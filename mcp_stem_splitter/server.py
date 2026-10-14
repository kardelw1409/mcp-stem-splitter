from __future__ import annotations

import logging
import os
import sys
import shutil
import subprocess
import zipfile
from pathlib import Path
import re
from typing import Annotated, Any, Literal, TypedDict

from mcp.server.fastmcp import FastMCP
from pydantic import Field

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

@mcp.tool("split_stems")
def split_stems(
    input_path: Annotated[str, Field(description="Audio file to split.")],
    output_dir: Annotated[str, Field(description="Directory for the stems; created if missing.")],
    model: Annotated[str, Field(description='Demucs model name: mdx_extra (default, best quality), htdemucs, htdemucs_ft (4x slower, ~1 % better), or htdemucs_6s.')] = "mdx_extra",
    stems: Annotated[str, Field(description="Only '4stems' is supported: drums, bass, vocals, other.")] = "4stems",
    audio_format: Annotated[str, Field(description="wav | flac | mp3 for the written stems.")] = "wav",
    keep_intermediates: Annotated[bool, Field(description="Keep intermediate files written by the separator.")] = False,
    filename_mode: Annotated[str, Field(description="fixed -> drums.wav; prefixed -> <inputname>_drums.wav.")] = "fixed",
    device: Annotated[str, Field(description="auto | cpu | cuda | cuda:N | directml. auto uses CUDA when the installed torch supports it.")] = "auto",
) -> dict[str, Any]:
    """Split a track into four stems: drums, bass, vocals, other.

    Use this when the user wants instrument stems for mixing, remixing or for feeding a
    later step (drum transcription needs the drum elements, which are a different split -
    see drums_midi). Weights download automatically on first use (~300 MB for mdx_extra).

    Returns absolute paths for every stem. On a 8 GB laptop GPU expect roughly 18 s per
    4 minutes of audio with htdemucs; mdx_extra is slower but separates better.
    """
    return split_stems_core(
        input_path=input_path,
        output_dir=output_dir,
        model=model,
        stems=stems,
        audio_format=audio_format,
        keep_intermediates=keep_intermediates,
        filename_mode=filename_mode,
        device=device,
    )


def split_stems_core(
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


@mcp.tool("list_models")
def list_models() -> dict[str, Any]:
    """List the supported separation models and presets.

    Call this before split_stems when unsure which model name is valid. The list is
    curated and short on purpose: names outside it are usually quantized variants that
    need C++ build tools on Windows.
    """
    return list_models_core()


def list_models_core() -> dict[str, Any]:
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


@mcp.tool("get_presets")
def get_presets() -> dict[str, Any]:
    """Describe the available stem presets (which outputs each one produces).

    Useful before split_stems to check whether a two-stem (vocals/instrumental) or a
    four-stem result is what the task needs.
    """
    return get_presets_core()


def get_presets_core() -> dict[str, Any]:
    _configure_logging()
    return {"presets": _get_presets()}


@mcp.tool("split_vocals_only")
def split_vocals_only(
    input_path: Annotated[str, Field(description="Audio file to split.")],
    output_dir: Annotated[str, Field(description="Directory for the two files; created if missing.")],
    model: Annotated[str, Field(description="Demucs model name; mdx_extra by default.")] = "mdx_extra",
    audio_format: Annotated[str, Field(description="wav | flac | mp3.")] = "wav",
    keep_intermediates: Annotated[bool, Field(description="Keep intermediate files.")] = False,
) -> dict[str, Any]:
    """Split a track into vocals + instrumental only (two files).

    Cheaper and cleaner than split_stems when nothing but a vocal/instrumental pair is
    needed, e.g. for karaoke or for a vocal-only analysis. The instrumental is the sum of
    the drums, bass and other stems that the model produced.
    """
    return split_vocals_only_core(
        input_path=input_path,
        output_dir=output_dir,
        model=model,
        audio_format=audio_format,
        keep_intermediates=keep_intermediates,
    )


def split_vocals_only_core(
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


@mcp.tool("drums_midi")
def drums_midi(
    audio_path: Annotated[str | None, Field(description="Audio to transcribe with ADTOF (no separation needed). At least one of audio_path / extra_midi / element_stems_dir must be given.")] = None,
    output_dir: Annotated[str, Field(description="Directory for the MIDI files; created if missing.")] = "drums_midi",
    element_stems_dir: Annotated[str | None, Field(description="Folder with the separated drum elements (kick/snare/hh/toms/ride/crash) of the SAME audio. They are the arbiter of every note and the source of velocity.")] = None,
    extra_midi: Annotated[list[str] | None, Field(description="Additional drum MIDI files to vote alongside ADTOF, e.g. the output of a second transcription model.")] = None,
    bpm: Annotated[float | None, Field(description="Tempo for 1/16 quantisation (use the MEASURED tempo). Omit to skip quantisation entirely.")] = None,
    grid_origin_s: Annotated[float | None, Field(description="Measured grid phase in seconds (e.g. grid.origin_s from the audiocpp manifest). Prefer this over estimating from the notes: an estimate cannot fix a systematic offset.")] = None,
    mode: Annotated[str, Field(description="balanced (default) | strict | consensus | full - how strictly sources must agree and the element support the note.")] = "balanced",
    adt_of_thresholds: Annotated[list[float] | None, Field(description="Per-class onset thresholds for ADTOF, in class order [kick, snare, tom, hihat, cymbal]. Omit for the model defaults; the measured set [0.176, 0.192, 0.32, 0.132, 0.24] trades a little precision for more hits (validated with the element support test).")] = None,
    allow_model_download: Annotated[bool, Field(description="Allow fetching the transcription weights once if they are not bundled. Set false for strictly offline runs.")] = True,
    quantize_to_grid: Annotated[bool, Field(description="Pull notes onto the 1/16 grid when bpm is given.")] = True,
    max_shift_ms: Annotated[float, Field(description="How far quantisation may move a note; anything further away is left untouched.")] = 60.0,
    split_by_class: Annotated[bool, Field(description="Also write one MIDI file per drum class (kick/snare/hihat/tom/cymbal) so a DAW can place them as separate items.")] = True,
    use_detector: Annotated[bool, Field(description="Also detect onsets inside the elements and let them vote. Turn off for the most conservative result.")] = True,
    detector_classes: Annotated[list[str] | None, Field(description='Restrict the detector to these classes, e.g. ["tom","cymbal"]. Recommended: a 5-class model has almost no toms/cymbals (16 and 6 over a 270 s track), while an unrestricted detector roughly doubles total density.')] = None,
    device: Annotated[str, Field(description="Torch device for the transcription model: auto | cpu | cuda | cuda:N.")] = "auto",
) -> dict[str, Any]:
    """Transcribe a track's drum part to MIDI - one file per drum class plus a combined file.

    Use this when the user wants drum MIDI, wants the drum part as separate items/tracks,
    or says the drum part is missing or does not sit on the grid. This tool does NOT
    separate audio: pass the already-separated drum elements in element_stems_dir.

    HOW A NOTE IS ACCEPTED
    Sources vote; the separated elements arbitrate.
      * audio_path is transcribed by ADTOF (a specialist drum-transcription model that
        works on the mix and uses its own per-class thresholds); measured cost is about
        2-4 s on GPU and up to ~10 s on CPU per 90 s of audio.
      * extra_midi files vote as well (any number of sources, each optional).
      * unless use_detector=False, onsets are also detected inside the elements.
      * a note must then be SUPPORTED by the matching element: louder than that element's
        OWN median level (self-calibrating; no absolute threshold is invented). Two
        sources agreeing are accepted from ~1.5 dB above that median; a lone source needs
        a clearer margin (~3 dB).
    Measured reason for this shape: one model's notes are only ~49 % supported by the
    audio, and even four models agreeing leave ~24 % unsupported - they share a training
    bias. The elements are the only physical reference that is independent of the models.
    Without element_stems_dir there is nothing to verify against; the report then says so
    (velocity_reference.source is "none") instead of pretending the notes were checked.

    MODES (precision vs recall)
      consensus - any 2+ sources agree, no physical check
      strict    - 2+ agree AND the element supports it
      balanced  - as strict, or a lone source clearly above the element level (default)
      full      - anything the element supports (recall first)
    Without element_stems_dir there is nothing to verify against: "full" then keeps every
    proposed note and "strict" keeps only agreed ones - the mode switches from filtering
    on physics to filtering on votes alone, which the report states explicitly.
    Note: strict needs two independent sources, so with a single model plus the detector
    it mainly keeps hi-hats. It is not a synonym for "better".

    MODEL ASSETS - NOTHING TO INSTALL
    The transcription model and its weights ship inside this server package, so the tool
    works on a fresh machine with no external download and no configured path. The file
    is matched against a pinned SHA-256 before loading; a wrong or half-copied blob is
    refused instead of silently loading random weights. Resolution order, first hit wins:
    MCP_STEM_SPLITTER_ADTOF_WEIGHTS, the bundled file, a cache directory, then a one-time
    download (only if allow_model_download is true). A GPU is used when present, otherwise
    CPU - the tool reports which in the result's model field rather than assuming.

    WHICH SOURCE COVERS WHICH CLASS (measured, full 270 s drum & bass track)
    A 5-class transcription model has almost no toms and cymbals (16 and 6 notes over the
    whole track - its vocabulary has no ride/crash), while the detector is productive
    exactly there. Recommended hybrid: detector_classes=["tom", "cymbal"] -> 2796 notes
    (kick 110, snare 227, hihat 730 from the model; tom 565, cymbal 1164 from the
    detector), against 4140 notes with an unrestricted detector and 1089 with none.
    Density matters musically: 4140 notes is ~22 per bar, which no drummer plays.

    GRID
    With bpm set, notes are pulled to the nearest 1/16 within max_shift_ms. Pass
    grid_origin_s whenever a measured grid phase exists - estimating it from the notes
    cannot correct a systematic offset and degenerates on short files. The report states
    which was used (grid.origin_source: "explicit" or "estimated_from_notes").

    VELOCITY IS PER CLASS (the models are not equally good at it)
      kick   - flat 100: model velocity correlates +0.09 with the audio (noise); a kick
               is a single-velocity sample trigger
      snare  - measured 88..124 (correlation +0.65 - real dynamics, e.g. ghost notes)
      hihat  - measured 36..127 (+0.45; carries the groove)
      tom / cymbal - measured 70..120 / 84..120 (compressed: less reliable)
    Dynamics come from the element's energy at each note, so they require
    element_stems_dir; otherwise every class gets one neutral value.

    ELEMENT FILES
    element_stems_dir must hold the elements of the SAME audio: kick, snare, hh, toms,
    ride, crash. Both kick.wav and drumsep_(kick)_....wav style names are recognised;
    cymbal sums ride+crash. A missing element costs only that class and is reported.

    OUTPUT
    <output_dir>/drums.mid plus drums_kick.mid, drums_snare.mid, drums_hihat.mid,
    drums_tom.mid, drums_cymbal.mid - each a GM drum track (kick 36, snare 38, hihat 42,
    tom 45, cymbal 49) so the DAW can place them as separate items.
    The result also carries notes{total,by_class,clusters_considered},
    velocity{...}, velocity_reference, grid{...}, arbiter{elements_used,durations,...},
    sources, detector_classes and model (which weights file was used, where it came
    from, its digest, and the device).

    ALWAYS READ warnings BEFORE REPORTING SUCCESS: they flag a missing element, a source
    that could not be read, and elements shorter than the audio (in which case every note
    past their end is dropped - the stems and the mix are not from the same file).
    ok=false means no source produced a single note.
    """
    from .drums import drums_midi_core

    return drums_midi_core(
        audio_path=audio_path,
        output_dir=output_dir,
        element_stems_dir=element_stems_dir,
        extra_midi=extra_midi,
        bpm=bpm,
        grid_origin_s=grid_origin_s,
        mode=mode,
        adt_of_thresholds=adt_of_thresholds,
        allow_model_download=allow_model_download,
        quantize_to_grid=quantize_to_grid,
        max_shift_ms=max_shift_ms,
        split_by_class=split_by_class,
        use_detector=use_detector,
        detector_classes=detector_classes,
        device=device,
    )


def run() -> None:
    _configure_logging()
    mcp.run()
