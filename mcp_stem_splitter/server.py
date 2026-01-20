from __future__ import annotations

import logging
import os
import sys
import shutil
import subprocess
import zipfile
from pathlib import Path
import re
from typing import Any, Literal, TypedDict

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("mcp_stem_splitter")

mcp = FastMCP("stem_splitter")

class Preset(TypedDict):
    name: str
    description: str
    outputs: list[str]


def _get_presets() -> list[Preset]:
    return [
        {
            "name": "2stems_vocals",
            "description": "Vocals + Instrumental (drums+bass+other)",
            "outputs": ["vocals", "instrumental"],
        },
        {
            "name": "4stems",
            "description": "Drums / Bass / Vocals / Other",
            "outputs": ["drums", "bass", "vocals", "other"],
        },
    ]


def _configure_logging() -> None:
    if logger.handlers:
        return
    logging.basicConfig(
        level=os.environ.get("STEM_SPLITTER_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _validate_input_file(input_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"input_path not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"input_path is not a file: {input_path}")


def _resolve_device(device: str) -> Any:
    import torch

    if device == "auto":
        if getattr(torch.version, "cuda", None) is None:
            logger.info("PyTorch build appears CPU-only (torch.version.cuda is None); using CPU.")
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "directml":
        try:
            import torch_directml  # type: ignore
        except Exception as e:
            raise RuntimeError("device='directml' requested but torch-directml is not installed.") from e
        return torch_directml.device()
    return device


def _write_source_audio(
    *,
    out_path: Path,
    source: Any,
    samplerate: int,
    audio_format: Literal["wav", "flac", "mp3"],
) -> None:
    import numpy as np
    import torch
    import soundfile as sf
    from demucs.audio import encode_mp3

    audio = source.detach().cpu().float().clamp_(-1.0, 1.0).numpy()
    audio = np.transpose(audio, (1, 0))  # [T, C]

    if audio_format == "mp3":
        encode_mp3(
            torch.from_numpy(audio).t(),
            out_path,
            samplerate=samplerate,
            bitrate=320,
            quality=2,
        )
        return

    sf.write(out_path, audio, samplerate, format=audio_format.upper())


def _demucs_separate(
    *,
    input_path: Path,
    output_dir: Path,
    model: str,
    audio_format: Literal["wav", "flac", "mp3"],
    device: str,
) -> tuple[Any, list[str], int, Literal["wav", "flac", "mp3"]]:
    """
    In-process Demucs separation that avoids torchaudio.save (which may require
    TorchCodec + FFmpeg shared DLLs on Windows).
    """
    import torch

    from demucs.apply import apply_model
    from demucs.audio import AudioFile
    from demucs.pretrained import get_model

    _ensure_dir(output_dir)

    # Ensure `ffmpeg` + `ffprobe` are available for demucs.audio.AudioFile.
    env = os.environ.copy()
    _ensure_ffmpeg_in_env(env)
    os.environ["PATH"] = env["PATH"]

    logger.info("Loading Demucs model: %s", model)
    demucs_model = get_model(model)
    demucs_model.cpu()
    demucs_model.eval()

    logger.info("Reading input audio via ffmpeg: %s", input_path)
    wav = AudioFile(input_path).read(streams=0, samplerate=demucs_model.samplerate, channels=demucs_model.audio_channels)

    ref = wav.mean(0)
    wav = wav - ref.mean()
    wav = wav / (ref.std() + 1e-9)

    resolved_device = _resolve_device(device)

    logger.info("Separating with device=%s", resolved_device)
    sources = apply_model(
        demucs_model,
        wav[None],
        device=resolved_device,
        shifts=1,
        split=True,
        overlap=0.25,
        progress=False,
        num_workers=0,
        segment=None,
    )[0]

    sources = sources * ref.std()
    sources = sources + ref.mean()

    return sources, list(demucs_model.sources), int(demucs_model.samplerate), audio_format


def _demucs_split_4stems(
    *,
    input_path: Path,
    output_dir: Path,
    model: str,
    audio_format: Literal["wav", "flac", "mp3"],
    filename_mode: Literal["fixed", "prefixed"],
    device: str,
) -> dict[str, str]:
    sources, source_names, samplerate, audio_format = _demucs_separate(
        input_path=input_path,
        output_dir=output_dir,
        model=model,
        audio_format=audio_format,
        device=device,
    )

    stems_out: dict[str, str] = {}
    safe_prefix = _safe_filename_prefix(input_path.stem)
    for source, stem_name in zip(sources, source_names):
        if stem_name not in {"drums", "bass", "vocals", "other"}:
            continue
        if filename_mode == "prefixed":
            out_name = f"{safe_prefix}_{stem_name}.{audio_format}"
        else:
            out_name = f"{stem_name}.{audio_format}"
        out_path = (output_dir / out_name).resolve()

        _write_source_audio(out_path=out_path, source=source, samplerate=samplerate, audio_format=audio_format)

        stems_out[stem_name] = str(out_path)

    missing = [s for s in ("drums", "bass", "vocals", "other") if s not in stems_out]
    if missing:
        raise RuntimeError(f"Demucs did not produce expected stems: {missing}")
    return stems_out


def _safe_filename_prefix(prefix: str) -> str:
    prefix = re.sub(r'[<>:"/\\\\|?*]+', "_", prefix)
    prefix = re.sub(r"\\s+", " ", prefix).strip()
    return prefix[:120] or "track"


def _ensure_ffmpeg_in_env(env: dict[str, str]) -> None:
    """
    demucs uses both `ffprobe` and `ffmpeg` executables. If they are not available,
    attempt to download a static Windows build to a local folder and prepend it to PATH.
    """
    local_bin = Path(__file__).resolve().parent.parent / ".ffmpeg"
    local_bin.mkdir(parents=True, exist_ok=True)

    ffmpeg_exe = local_bin / "ffmpeg.exe"
    ffprobe_exe = local_bin / "ffprobe.exe"

    if not (ffmpeg_exe.exists() and ffprobe_exe.exists()):
        _download_ffmpeg_essentials(local_bin)

    env["PATH"] = str(local_bin) + os.pathsep + env.get("PATH", "")


def _download_ffmpeg_essentials(local_bin: Path) -> None:
    import urllib.request

    url = os.environ.get(
        "STEM_SPLITTER_FFMPEG_ZIP_URL",
        "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    )
    zip_path = local_bin / "ffmpeg-release-essentials.zip"
    logger.info("Downloading FFmpeg (essentials) to %s", zip_path)
    urllib.request.urlretrieve(url, zip_path)  # nosec - controlled URL

    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        ffmpeg_member = next((m for m in members if m.endswith("/bin/ffmpeg.exe")), None)
        ffprobe_member = next((m for m in members if m.endswith("/bin/ffprobe.exe")), None)
        if not ffmpeg_member or not ffprobe_member:
            raise RuntimeError("FFmpeg zip did not contain expected ffmpeg.exe/ffprobe.exe under /bin/")

        zf.extract(ffmpeg_member, local_bin)
        zf.extract(ffprobe_member, local_bin)

        extracted_ffmpeg = local_bin / ffmpeg_member
        extracted_ffprobe = local_bin / ffprobe_member
        shutil.copyfile(extracted_ffmpeg, local_bin / "ffmpeg.exe")
        shutil.copyfile(extracted_ffprobe, local_bin / "ffprobe.exe")

    try:
        zip_path.unlink(missing_ok=True)  # type: ignore[call-arg]
    except Exception:
        pass

@mcp.tool()
def split_stems(
    input_path: str,
    output_dir: str,
    model: str = "mdx_extra",
    stems: str = "4stems",
    audio_format: str = "wav",
    keep_intermediates: bool = False,
    filename_mode: str = "fixed",
    device: str = "auto",
) -> dict[str, Any]:
    """
    Split an audio file into 4 stems (drums/bass/vocals/other) and return absolute paths.
    """
    _configure_logging()

    if stems != "4stems":
        raise ValueError("Only stems='4stems' is currently supported.")
    if audio_format not in {"wav", "flac", "mp3"}:
        raise ValueError("audio_format must be one of: wav, flac, mp3")
    if filename_mode not in {"fixed", "prefixed"}:
        raise ValueError("filename_mode must be one of: fixed, prefixed")
    if device not in {"auto", "cpu", "cuda", "directml"} and not device.startswith("cuda:"):
        raise ValueError("device must be one of: auto, cpu, cuda, cuda:N, directml")
    if model.endswith("_q"):
        raise ValueError(
            "Quantized demucs models (e.g. '*_q') require the optional 'diffq' dependency, "
            "which typically needs C++ build tools on Windows. Use a non-quantized model "
            "like 'mdx_extra' or 'htdemucs' instead."
        )

    input_path_p = Path(input_path).expanduser()
    output_dir_p = Path(output_dir).expanduser()
    _validate_input_file(input_path_p)
    _ensure_dir(output_dir_p)

    stems_out = _demucs_split_4stems(
        input_path=input_path_p,
        output_dir=output_dir_p,
        model=model,
        audio_format=audio_format,  # type: ignore[arg-type]
        filename_mode=filename_mode,  # type: ignore[arg-type]
        device=device,
    )

    return {
        "input": str(input_path_p.resolve()),
        "output_dir": str(output_dir_p.resolve()),
        "model": model,
        "device": device,
        "stems": stems_out,
    }


@mcp.tool()
def list_models() -> dict[str, Any]:
    _configure_logging()

    models = [
        "htdemucs",
        "htdemucs_ft",
        "mdx_extra",
    ]
    return {
        "models": models,
        "presets": [p["name"] for p in _get_presets()],
        "notes": (
            "Model list is curated (Demucs will download weights on first use). "
            "Supported presets: 4stems and 2stems_vocals."
        ),
    }


@mcp.tool()
def get_presets() -> dict[str, Any]:
    _configure_logging()
    return {"presets": _get_presets()}


@mcp.tool()
def split_vocals_only(
    input_path: str,
    output_dir: str,
    model: str = "mdx_extra",
    audio_format: str = "wav",
    keep_intermediates: bool = False,
) -> dict[str, Any]:
    """
    Split an audio file into vocals + instrumental and return absolute paths.
    """
    _configure_logging()

    if audio_format not in {"wav", "flac", "mp3"}:
        raise ValueError("audio_format must be one of: wav, flac, mp3")
    if model.endswith("_q"):
        raise ValueError(
            "Quantized demucs models (e.g. '*_q') require the optional 'diffq' dependency, "
            "which typically needs C++ build tools on Windows. Use a non-quantized model "
            "like 'mdx_extra' or 'htdemucs' instead."
        )

    input_path_p = Path(input_path).expanduser()
    output_dir_p = Path(output_dir).expanduser()
    _validate_input_file(input_path_p)
    _ensure_dir(output_dir_p)

    sources, source_names, samplerate, audio_format = _demucs_separate(
        input_path=input_path_p,
        output_dir=output_dir_p,
        model=model,
        audio_format=audio_format,  # type: ignore[arg-type]
        device="auto",
    )

    vocals_idx = None
    for idx, name in enumerate(source_names):
        if name == "vocals":
            vocals_idx = idx
            break
    if vocals_idx is None:
        raise RuntimeError("Demucs model did not provide a 'vocals' stem.")

    vocals = sources[vocals_idx]
    instrumental = None
    for idx, name in enumerate(source_names):
        if idx == vocals_idx:
            continue
        if name in {"drums", "bass", "other"}:
            instrumental = sources[idx] if instrumental is None else (instrumental + sources[idx])

    if instrumental is None:
        raise RuntimeError("Could not build instrumental stem from model outputs.")

    vocals_path = (output_dir_p / f"vocals.{audio_format}").resolve()
    instrumental_path = (output_dir_p / f"instrumental.{audio_format}").resolve()
    _write_source_audio(out_path=vocals_path, source=vocals, samplerate=samplerate, audio_format=audio_format)
    _write_source_audio(out_path=instrumental_path, source=instrumental, samplerate=samplerate, audio_format=audio_format)

    return {
        "input": str(input_path_p.resolve()),
        "output_dir": str(output_dir_p.resolve()),
        "model": model,
        "stems": {
            "vocals": str(vocals_path),
            "instrumental": str(instrumental_path),
        },
    }


def run() -> None:
    _configure_logging()
    mcp.run()
