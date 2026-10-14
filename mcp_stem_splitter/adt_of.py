"""ADTOF drum transcription, made portable.

The model is vendored under :mod:`mcp_stem_splitter.vendor.adtof_pytorch` (four small
files) together with its weight file, so this MCP does not depend on a checkout path on
anyone's disk: install it anywhere, on any machine, and it runs offline. If the weights
are missing (a slim install), they are fetched once into a cache directory and verified
by SHA-256 before use.

Two traps in the upstream code are handled here instead of being left to the caller:

* upstream loads the weights only ``if weights_path is not None and weights_path.exists()``
  - so a missing file silently yields RANDOM weights and a plausible-looking MIDI. We
  refuse to run in that case and always report which weights file was used.
* upstream prints ``"Loading PyTorch weights from: ..."`` to stdout. Under an MCP stdio
  transport stdout IS the JSON-RPC channel, so that print corrupts the protocol. Calls
  made through this module send library stdout to stderr.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Callable, Sequence

logger = logging.getLogger("mcp_stem_splitter")

WEIGHTS_FILENAME = "adtof_frame_rnn_pytorch_weights.pth"
# sha256 of the file vendored under vendor/adtof_pytorch/data/ (verified at build time);
# the same constant gates the download, so a tampered or truncated transfer is rejected.
WEIGHTS_SHA256 = "1bc986e596ec47ba0b44916f87cd4a39f0b2bec23596df3fb5d0e87749217320"
WEIGHTS_URL = (
    "https://raw.githubusercontent.com/xavriley/ADTOF-pytorch/main/"
    f"src/adtof_pytorch/data/{WEIGHTS_FILENAME}"
)
ENV_WEIGHTS = "MCP_STEM_SPLITTER_ADTOF_WEIGHTS"
ENV_CACHE = "MCP_STEM_SPLITTER_CACHE"


class WeightsNotFound(RuntimeError):
    """Raised when no usable ADTOF weights file can be found or fetched."""


def cache_dir() -> Path:
    """Where downloaded models live. Overridable with MCP_STEM_SPLITTER_CACHE."""
    override = os.environ.get(ENV_CACHE)
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return Path(base) / "mcp-stem-splitter"
    return Path.home() / ".cache" / "mcp-stem-splitter"


def vendored_weights_path() -> Path | None:
    """The weights shipped inside this package, if present."""
    try:
        from importlib.resources import files

        candidate = Path(str(files(__package__) / "vendor" / "adtof_pytorch" / "data" / WEIGHTS_FILENAME))
        return candidate if candidate.is_file() else None
    except Exception:  # a namespace/zip install can defeat importlib.resources
        return None


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_weights(path: Path, expected: str | None = WEIGHTS_SHA256) -> str:
    """Return the file's sha256, raising if it does not match the expected one."""
    actual = sha256_file(path)
    if expected and actual != expected:
        raise WeightsNotFound(
            f"weights at {path} have sha256 {actual}, expected {expected}; "
            "delete the file to let it be fetched again"
        )
    return actual


def download_weights(destination: Path, url: str = WEIGHTS_URL, timeout: float = 120.0) -> Path:
    """Fetch the weights atomically: write a .part file, verify, then move into place."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    logger.info("downloading ADTOF weights from %s", url)
    with urllib.request.urlopen(url, timeout=timeout) as response, partial.open("wb") as handle:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            handle.write(block)
    try:
        verify_weights(partial)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    os.replace(partial, destination)
    return destination


def resolve_weights(allow_download: bool = True) -> tuple[Path, str]:
    """Find the weights: explicit env var, then vendored, then cache, then download.

    Returns (path, source) where source is one of "env", "vendored", "cache",
    "downloaded" - reported to the caller so a run can never be mistaken about which
    weights produced the MIDI.
    """
    override = os.environ.get(ENV_WEIGHTS)
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise WeightsNotFound(f"{ENV_WEIGHTS} points at a missing file: {path}")
        verify_weights(path)
        return path, "env"

    vendored = vendored_weights_path()
    if vendored is not None:
        verify_weights(vendored)
        return vendored, "vendored"

    cached = cache_dir() / WEIGHTS_FILENAME
    if cached.is_file():
        verify_weights(cached)
        return cached, "cache"

    if not allow_download:
        raise WeightsNotFound(
            f"no ADTOF weights found (looked in the package and in {cache_dir()}) and "
            "downloads are disabled for this call"
        )
    return download_weights(cached), "downloaded"


def load_transcribe_function() -> tuple[Callable[..., Any], str]:
    """Import the vendored model, falling back to an installed copy of the package."""
    try:
        from .vendor.adtof_pytorch import transcribe_to_midi  # type: ignore[attr-defined]

        return transcribe_to_midi, "vendored"
    except Exception as vendored_error:  # pragma: no cover - depends on environment
        logger.debug("vendored ADTOF unusable (%s); trying an installed copy", vendored_error)
        try:
            from adtof_pytorch import transcribe_to_midi  # type: ignore[import-not-found]

            return transcribe_to_midi, "installed"
        except Exception as installed_error:
            raise RuntimeError(
                f"ADTOF is unavailable: vendored import failed ({vendored_error}); "
                f"installed import failed ({installed_error})"
            ) from installed_error


def transcribe(
    audio_path: str | Path,
    midi_out: str | Path,
    *,
    thresholds: Sequence[float] | None = None,
    device: str = "cpu",
    allow_download: bool = True,
) -> dict[str, Any]:
    """Transcribe audio to a drum MIDI file, returning where the weights came from.

    `device` must be "cpu" or "cuda" (the upstream signature asserts this).
    """
    function, implementation = load_transcribe_function()
    weights_path, weights_source = resolve_weights(allow_download=allow_download)
    weights_sha = verify_weights(weights_path)

    # upstream asserts device in ("cuda", "cpu"), so "cuda:1" must be narrowed to "cuda"
    device = "cuda" if str(device).startswith("cuda") else "cpu"

    kwargs: dict[str, Any] = {"weights": str(weights_path), "device": device}
    if thresholds:
        kwargs["thresholds"] = [float(value) for value in thresholds]

    # library prints go to stderr: stdout carries the MCP JSON-RPC stream
    with contextlib.redirect_stdout(sys.stderr):
        function(str(audio_path), str(midi_out), **kwargs)

    if not Path(midi_out).is_file():
        raise RuntimeError(f"ADTOF did not write {midi_out}")

    return {
        "weights": str(weights_path),
        "weights_source": weights_source,
        "weights_sha256": weights_sha,
        "implementation": implementation,
        "device": device,
        "thresholds": list(thresholds) if thresholds else "model_defaults",
    }
