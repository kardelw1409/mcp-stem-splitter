"""Drum MIDI from audio: several sources vote, separated drum elements arbitrate.

The output is deliberately split per class (kick / snare / hihat / tom / cymbal) so a
DAW can place each part as its own item, plus one combined file.

Why not just one transcription model:

* a single model's notes are only ~49 % supported by the audio (measured),
* even four models agreeing leave ~24 % of notes unsupported - they share a training bias,
* the drum elements themselves (kick.wav, snare.wav, ...) are the only ground truth we
  physically have, and they are independent of every model.

So each note must both be proposed by a source and (when elements are available) be
supported by the matching element's own energy at that instant. The support test is
self-calibrating: a note must be louder than the element's own median level, so no
absolute threshold has to be invented.

Velocity is per class, because the models are not equally good at it:

=================  ======================  ===============================================
class              policy                  measured reason
=================  ======================  ===============================================
kick               flat (100)               corr(velocity, energy) = +0.09 -> guessing;
                                           a kick is a single-velocity sample trigger
snare              measured 88..124         corr = +0.65 -> real dynamics (ghost notes)
hihat              measured 36..127         corr = +0.45 -> real, and carries the groove
tom / cymbal       measured, narrow         models less reliable here; range compressed
=================  ======================  ===============================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("mcp_stem_splitter")

CLASSES = ("kick", "snare", "hihat", "tom", "cymbal")

# canonical class -> (element names to sum for the arbiter/detector, analysis band)
ELEMENT_BANDS: dict[str, tuple[tuple[str, ...], tuple[float, float]]] = {
    # The kick's SUPPORT band is its body, not its sub.  Measured on a real stem, asking
    # "is there kick energy here" in 25-120 Hz gives +3.7 dB at the hits and +5.0 dB BETWEEN
    # them - i.e. it is louder where there is no hit, so the answer is never no.  In 60-250 Hz
    # the same question separates hits from the gaps by +7.2 dB, which is what lets the
    # arbiter refuse a wrong note at all.
    "kick": (("kick",), (60.0, 250.0)),
    "snare": (("snare",), (150.0, 400.0)),
    "hihat": (("hh",), (6000.0, 12000.0)),
    "tom": (("toms",), (80.0, 250.0)),
    "cymbal": (("ride", "crash"), (4000.0, 14000.0)),
}

# GM note -> canonical class. Covers both vocabularies in play: ADTOF's 5 labels use
# GM 35 for kick, ADT_STR emits 36 plus a 26-class map (comparing raw pitches once made
# ADT_STR look like it found zero kicks - that was the metric's fault, not the model's).
PITCH_CLASS: dict[int, str] = {
    35: "kick", 36: "kick",
    37: "snare", 38: "snare", 39: "snare", 40: "snare",
    41: "tom", 43: "tom", 45: "tom", 47: "tom", 48: "tom", 50: "tom",
    42: "hihat", 44: "hihat", 46: "hihat",
    49: "cymbal", 51: "cymbal", 52: "cymbal", 53: "cymbal", 55: "cymbal",
    57: "cymbal", 59: "cymbal",
}

# written back to the output: one GM note per class
DEFAULT_OUTPUT_PITCH = {"kick": 36, "snare": 38, "hihat": 42, "tom": 45, "cymbal": 49}

# percussion that is not part of a metal/neurofunk kit; models hallucinate these on
# unfamiliar material (they were 22 % of ADT_STR's output on our drum & bass track)
OFF_KIT_PITCHES = {54, 56, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
                   73, 74, 75, 76, 77, 78, 79, 80, 81}

VELOCITY_POLICY: dict[str, tuple[str, int, int]] = {
    "kick": ("flat", 100, 100),
    "snare": ("measured", 88, 124),
    "hihat": ("measured", 36, 127),
    "tom": ("measured", 70, 120),
    "cymbal": ("measured", 84, 120),
}

MODES = ("balanced", "strict", "consensus", "full")


# --------------------------------------------------------------------------------------
# element stems: discovery, band energy, onsets, support
# --------------------------------------------------------------------------------------


def find_element_wav(element_dir: Path, names: Iterable[str]) -> Path | None:
    """Locate e.g. kick.wav or drumsep_(kick)_....wav inside a folder."""
    for name in names:
        for pattern in (f"{name}.wav", f"*_({name})_*.wav", f"*({name})*.wav", f"*_{name}.wav", f"*{name}*.wav"):
            hits = sorted(element_dir.glob(pattern))
            if hits:
                return hits[0]
    return None


def _band_signal(path: Path, lo: float, hi: float):
    """Band-limited version of a file, read at ITS OWN sample rate (never the mix's)."""
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    spectrum = np.fft.rfft(mono.astype(np.float64))
    freqs = np.fft.rfftfreq(len(mono), 1.0 / sr)
    band = np.fft.irfft(spectrum * ((freqs >= lo) & (freqs <= hi)), n=len(mono))
    return np.asarray(band, dtype=np.float32), int(sr)


# The band a class is TIMED in is not always the band it is SUPPORTED in, and the kick is
# why.  A kick's 25-120 Hz is where its weight is, but it is also where it RINGS: the
# stem's low band is busy between hits, so a hit's onset is wherever a threshold happens to
# fall.  Measured on a real DrumSep kick stem (270 s) with librosa's detector, moving ONE
# constant (the refractory) from 2 to 8 frames:
#
#     25-120 Hz  count 2065 -> 1717 -> 1281   (46 % spread)
#     60-250 Hz  count  954 ->  918 ->  890   ( 7 % spread)
#    120-2k Hz   count  654 ->  650 ->  638   ( 2 % spread)
#
# and the pipeline's own detector, run in 25-120 Hz, produced 620 onsets where the same
# algorithm with a 150 ms refractory produced 427 - two settings agreeing on only 85 of
# them.  A number that moves 46 % when a constant is nudged is a property of the constant,
# not of the drum.  The kick is therefore timed in its body band, where the count is a
# measurement.
DETECT_BANDS: dict[str, tuple[float, float]] = {
    "kick": (60.0, 250.0),
    "snare": (150.0, 400.0),
    "hihat": (6000.0, 12000.0),
    "tom": (80.0, 250.0),
    "cymbal": (4000.0, 14000.0),
}

# A refractory shorter than the band's own ring period cannot separate hits from ringing:
# the lowest frequency the band admits repeats every `1 / lo` seconds, so anything above
# three cycles of that is a new hit by construction.  With the kick's OLD band this rule
# asks for 120 ms against the 60 ms floor the code used - which is the other half of the
# same defect.  It is a rule, not a tuned number, and it is what keeps the count stable.
MIN_REFRACTORY_S = 0.060

# Below this contrast (p90 - p10 of the band's own level, in dB) a class's onsets are
# threshold-dependent, and the report says so instead of presenting them as measurements.
LOW_CONTRAST_DB = 12.0


def refractory_for_band(lo_hz: float) -> float:
    """The shortest gap in which a band cannot produce two independent hits."""
    return max(MIN_REFRACTORY_S, 3.0 / max(lo_hz, 1.0))


def _band_signal_pair(path: Path, support: tuple[float, float], detect: tuple[float, float]):
    """Both bands of one file from ONE read + ONE fft (a stem is 270 s of float32)."""
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1).astype(np.float64)
    spectrum = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(len(mono), 1.0 / sr)
    out = []
    for lo, hi in (support, detect):
        mask = (freqs >= lo) & (freqs <= hi)
        out.append(np.asarray(np.fft.irfft(spectrum * mask, n=len(mono)), dtype=np.float32))
    return out[0], out[1], int(sr)


def _element_signal(element_dir: Path, cls: str):
    """Sum of the class's element files + its own median level (the support reference).

    Carries BOTH bands: `band` for the support test (is there energy of this class here)
    and `detect_band` for onset timing (see DETECT_BANDS).
    """
    import numpy as np

    names, support = ELEMENT_BANDS[cls]
    detect = DETECT_BANDS.get(cls, support)
    total = None
    detect_total = None
    sr = None
    used: list[str] = []
    for name in names:
        path = find_element_wav(element_dir, [name])
        if path is None:
            continue
        band, detect_band, sr = _band_signal_pair(path, support, detect)
        total = band if total is None else total[: len(band)] + band[: len(total)]
        if detect_total is None:
            detect_total = detect_band
        else:
            detect_total = detect_total[: len(detect_band)] + detect_band[: len(detect_total)]
        used.append(path.name)
    if total is None:
        return None
    step = max(1, int(0.025 * sr))
    frames = total[: len(total) // step * step].reshape(-1, step)
    env = np.sqrt(np.mean(frames ** 2, axis=1))
    env_db = 20.0 * np.log10(np.maximum(env, 1e-12))
    p10, p90 = (float(np.percentile(env_db, p)) for p in (10.0, 90.0))
    return {
        "band": total,
        "detect_band": detect_total,
        "sr": int(sr),
        "background": float(np.median(env)),
        "files": used,
        "detect_band_hz": [float(detect[0]), float(detect[1])],
        # contrast = how far the quiet parts sit below the loud parts.  A band whose hits
        # do not stand above a busy floor cannot be timed by thresholding it, and this is
        # the number that says so instead of trusting the count.
        "contrast_db": round(p90 - p10, 2),
    }


def _support_db(signal: dict, t: float, half: float = 0.03) -> float:
    import numpy as np

    band, sr = signal["band"], signal["sr"]
    a, b = max(0, int((t - half) * sr)), int((t + half) * sr)
    chunk = band[a:b]
    if len(chunk) == 0 or signal["background"] <= 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(chunk ** 2)))
    return float(20 * np.log10(max(rms, 1e-12) / signal["background"]))


# Onset detection constants.  The threshold is LOCAL (a multiple of the flux's own level
# in a short window) and not a percentile of the whole file: a file-wide constant cannot be
# right in a quiet passage and a loud one at once.  Measured on a real kick stem, moving
# only the refractory: the whole-file version returned 467/382/296/268 onsets (56 % spread,
# 57 % of the intervals landing on a musical subdivision) where the local version returns
# 381/370/341/289 (27 % spread, 92 % musical).  Intervals are the tell: ringing inside one
# hit produces intervals that are not subdivisions of anything.
#
# Against an independent reference (the envelope's own strong peaks) the two rules score:
#
#     whole-file percentile   recall 40 %  precision 34 %   <- what shipped
#     local x7 only           recall 69 %  precision 47 %   <- fires in silence (6 %)
#     local x7 + floor        recall 57 %  precision 54 %   <- both terms
ONSET_FACTOR = 7.0        # how far the flux must stand above its own local level
# The floor's job is to keep the material's residue out (reverb tails, separation bleed):
# with the local term alone, 6 % of the onsets sat more than 10 dB below the file's median,
# i.e. where nothing was hit.  Its price is measured too: a hit more than ~12 dB below the
# track's strongest is not reported by the detector (the model can still contribute it).
ONSET_FLOOR_FRACTION = 0.2
# The floor's percentile must be HIGH: a percentile is only a hit-level number if the hits
# occupy more than 1-p of the signal.  At p99 a file whose hits are sparse has a floor made
# of its own noise - measured on a synthetic 6 s file with five clicks, which then produced
# onsets in pure silence.  p99.9 is a hit either way: on a real 270 s stem the two differ by
# a factor of 2, and the choice costs nothing there.
ONSET_FLOOR_PERCENTILE = 99.9
ONSET_WINDOW_S = 0.200    # the window that level is measured over
ONSET_FRAME_S = 0.002     # analysis hop for the flux (finer than any grid a note lands on)


def _onset_times(band, sr: int, min_gap: float = 0.06) -> list[float]:
    """Attack times from the band-limited envelope, thresholded LOCALLY.

    `min_gap` is the refractory (see refractory_for_band): it must outlast the band's own
    ring, or the ring's bumps become notes.  It is a guard, not the mechanism - with a
    local threshold the flux does not oscillate through it, so the count does not collapse
    when it is moved.
    """
    import numpy as np

    window = max(int(0.005 * sr), 1)
    env = np.convolve(np.abs(band), np.ones(window) / window, mode="same")
    flux = np.diff(env)
    flux[flux < 0] = 0.0

    hop = max(int(ONSET_FRAME_S * sr), 1)
    frames = flux[::hop]
    n = len(frames)
    if n < 8:
        return []
    span = max(int(ONSET_WINDOW_S * sr / hop), 3)
    pad = span // 2
    padded = np.pad(frames, (pad, pad), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, span)
    stride = max(1, span // 8)
    sampled = windows[::stride].mean(axis=1)
    level = sampled[np.linspace(0, len(sampled) - 1, n).astype(int)]
    # BOTH terms are needed, and each was measured.  The local term is what makes the same
    # setting work in a quiet passage and a loud one; alone it goes to zero in silence, so
    # tiny flux passes and the onsets land where nothing was hit (measured: 6 % of them sat
    # more than 10 dB below the file's median).  The floor alone is the old behaviour, which
    # let ring bumps through (65 % musical intervals, and only 79 % of its onsets at a real
    # rise).  Together: 86 % musical and 95 % at a rise, with nothing in the silence.
    threshold = np.maximum(
        level * ONSET_FACTOR,
        ONSET_FLOOR_FRACTION * float(np.percentile(flux, ONSET_FLOOR_PERCENTILE)),
    )

    above = frames > threshold
    candidates = np.where(above[1:-1] & (above[1:-1] >= above[:-2]) & (above[1:-1] > above[2:]))[0] + 1
    times: list[float] = []
    last = -10.0
    for index in candidates:
        # refine to the full-resolution flux so the reported time keeps its sub-frame detail
        lo = max(0, index * hop - hop)
        hi = min(len(flux), index * hop + hop + 1)
        exact = lo + int(np.argmax(flux[lo:hi]))
        t = exact / sr
        if t - last > min_gap:
            times.append(float(t))
            last = t
    return times


# --------------------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------------------


def _notes_from_midi(path: Path, label: str, warnings: list[str]) -> list[tuple[str, float]]:
    """Read a drum MIDI file and map its pitches onto the canonical classes."""
    import pretty_midi

    try:
        midi = pretty_midi.PrettyMIDI(str(path))
    except Exception as error:  # a broken source must not kill the whole fusion
        warnings.append(f"{label}: not readable as MIDI ({error})")
        return []

    notes: list[tuple[str, float]] = []
    off_kit = 0
    for instrument in midi.instruments:
        for note in instrument.notes:
            cls = PITCH_CLASS.get(note.pitch)
            if cls is None:
                if note.pitch in OFF_KIT_PITCHES:
                    off_kit += 1
                continue
            notes.append((cls, float(note.start)))
    if off_kit:
        logger.info("%s: dropped %d off-kit percussion notes", label, off_kit)
    return notes


def _notes_from_adt_of(
    path: Path,
    warnings: list[str],
    thresholds: list[float] | None = None,
    device: str = "auto",
    allow_download: bool = True,
) -> tuple[list[tuple[str, float]], dict]:
    """ADTOF drum transcription of an audio file, in a temp file, plus provenance.

    The model is reached through `adt_of`, which owns the weights (vendored in this
    package, or downloaded and checksum-verified) and keeps library prints off stdout -
    stdout carries the MCP protocol. Its provenance is returned so the caller can tell
    which weights produced the notes.
    """
    import tempfile

    from . import adt_of

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "adtof.mid"
        try:
            info = adt_of.transcribe(
                path,
                out,
                thresholds=thresholds,
                device=_resolve_torch_device(device),
                allow_download=allow_download,
            )
        except Exception as error:
            warnings.append(f"adt_of({path.name}): transcription unavailable ({error})")
            return [], {}
        logger.info(
            "adt_of %s: weights from %s (%s)",
            path.name,
            info["weights_source"],
            info["implementation"],
        )
        return _notes_from_midi(out, f"adt_of({path.name})", warnings), info


def _resolve_torch_device(device: str) -> str:
    try:
        import torch

        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return device
    except Exception:
        return "cpu"


def _notes_from_elements(signals: dict[str, dict], element_dir: Path, warnings: list[str]) -> list[tuple[str, float]]:
    """Independent onsets measured inside each element - recall, not authority.

    These exist because the models miss hits; but they are found in the same elements
    the arbiter uses, so they can never be evidence FOR themselves.

    Timed in the class's DETECT band with a refractory that outlasts that band's own ring
    (see DETECT_BANDS / refractory_for_band).  In a band whose hits do not stand above a
    busy floor BOTH the count and the times are properties of the constants: measured on a
    real kick stem, this same algorithm returned 620 or 427 onsets depending on one number,
    agreeing with itself on only 85 of them.

    Takes the signals the caller already computed: re-reading six 270 s stems to ask a
    second question about them is 400 MB of I/O for nothing.
    """
    notes: list[tuple[str, float]] = []
    for cls in CLASSES:
        signal = signals.get(cls)
        if signal is None:
            continue
        lo_hz, hi_hz = signal["detect_band_hz"]
        refractory = refractory_for_band(float(lo_hz))
        if signal["contrast_db"] < LOW_CONTRAST_DB:
            warnings.append(
                f"element '{cls}': its {lo_hz:.0f}-{hi_hz:.0f} Hz band has only "
                f"{signal['contrast_db']:.1f} dB between the quiet and the loud parts, so its "
                "onsets depend on the threshold - this class's note times are approximate"
            )
        for t in _onset_times(signal["detect_band"], signal["sr"], min_gap=refractory):
            notes.append((cls, float(t)))
    if not notes:
        warnings.append(f"detector: no drum elements found in {element_dir}")
    return notes


# --------------------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------------------


def _cluster(notes_by_source: dict[str, list[tuple[str, float]]], step: float) -> list[dict]:
    clusters: dict[tuple[str, int], dict] = {}
    for label, notes in notes_by_source.items():
        for cls, t in notes:
            key = (cls, int(round(t / step)))
            entry = clusters.setdefault(key, {"class": cls, "times": [], "sources": set()})
            entry["times"].append(t)
            entry["sources"].add(label)

    import numpy as np

    out = []
    for entry in clusters.values():
        out.append(
            {
                "class": entry["class"],
                "onset": float(np.median(entry["times"])),
                "votes": len(entry["sources"]),
                "sources": sorted(entry["sources"]),
            }
        )
    return sorted(out, key=lambda r: (r["class"], r["onset"]))


def _select(records: list[dict], signals: dict, mode: str, min_support_db: float, lone_support_db: float) -> list[dict]:
    """Keep a cluster if the sources agree, and the element backs it up when available."""
    kept: list[dict] = []
    for rec in records:
        signal = signals.get(rec["class"])
        rec["support_db"] = round(_support_db(signal, rec["onset"]), 2) if signal else None
        agreed = rec["votes"] >= 2
        supported = rec["support_db"] is None or rec["support_db"] >= min_support_db
        lone_ok = rec["support_db"] is None or rec["support_db"] >= lone_support_db

        if mode == "consensus":
            ok = agreed
        elif mode == "strict":
            ok = agreed and supported
        elif mode == "balanced":
            ok = (agreed and supported) or (not agreed and lone_ok)
        else:  # full
            ok = supported
        if ok:
            kept.append(rec)
    return kept


def _grid_origin(records: list[dict], step: float) -> float:
    """Where the 1/16 grid actually sits, measured from the notes themselves."""
    import numpy as np

    if not records:
        return 0.0
    offsets = np.array([(rec["onset"] / step) % 1.0 for rec in records])
    return float(np.angle(np.mean(np.exp(2j * np.pi * offsets))) / (2 * np.pi) * step)


def _quantize(records: list[dict], bpm: float | None, max_shift_s: float, grid_origin_s: float | None = None) -> tuple[list[dict], dict]:
    if not bpm:
        return records, {"applied": False, "reason": "no bpm given"}
    step = 60.0 / bpm / 4  # 1/16
    # The grid's phase must NOT be invented from the notes when the caller knows it:
    # estimating it from the notes (a) cannot fix a systematic offset, only de-jitter,
    # and (b) degenerates on a short file - a single note would define its own grid and
    # always "snap" to itself. The manifest already carries the measured phase, so it
    # can be passed in; only without it do we fall back to the notes' own mean phase.
    if grid_origin_s is not None:
        origin = float(grid_origin_s) % step
        origin_source = "explicit"
    else:
        origin = _grid_origin(records, step)
        origin_source = "estimated_from_notes"
    moved = 0
    snaps = 0
    out = []
    for rec in records:
        target = origin + round((rec["onset"] - origin) / step) * step
        shift = target - rec["onset"]
        if abs(shift) <= max_shift_s:
            rec = {**rec, "onset": max(0.0, target), "snap_ms": round(shift * 1000, 1)}
            snaps += 1
            moved += abs(shift) > 1e-6
        out.append(rec)
    return out, {
        "applied": True,
        "bpm": bpm,
        "sixteenth_s": round(step, 6),
        "origin_s": round(origin, 6),
        "origin_source": origin_source,
        "max_shift_ms": max_shift_s * 1000,
        "notes_snapped": snaps,
        "notes_moved": moved,
    }


def _assign_velocity(records: list[dict], signals: dict) -> None:
    import numpy as np

    by_class: dict[str, list[dict]] = {}
    for rec in records:
        by_class.setdefault(rec["class"], []).append(rec)

    for cls, recs in by_class.items():
        kind, lo, hi = VELOCITY_POLICY.get(cls, ("flat", 100, 100))
        if kind == "flat" or lo == hi:
            for rec in recs:
                rec["velocity"] = lo
            continue
        signal = signals.get(cls)
        levels = np.array([_support_db(signal, rec["onset"]) for rec in recs]) if signal else np.zeros(len(recs))
        spread = float(levels.max() - levels.min()) if len(levels) else 0.0
        if len(levels) > 2 and spread > 0.5:
            normalised = (levels - levels.min()) / spread
        else:
            normalised = np.full(len(levels), 0.5)
        for rec, value in zip(recs, normalised):
            rec["velocity"] = int(round(lo + (hi - lo) * float(value)))


def _write_midi(
    records: list[dict], path: Path, pitch_map: dict[str, int], bpm: float | None = None
) -> dict[str, Any]:
    """Write one drum part, carrying the PROJECT tempo when it is known.

    This is not cosmetic. A DAW plays a MIDI item's ticks as beats at the project
    tempo (`IGNTEMPO 0` in REAPER), so a file left at pretty_midi's default 120 in a
    172 bpm project plays 1.43x too fast and ends far early - measured on a real
    bundle: a 120 bpm drums.mid ended at 186.74 s of project time instead of
    267.66 s. Passing `initial_tempo` makes the seconds -> ticks conversion use the
    project tempo, which is the part a plain `set_tempo` afterwards cannot fix.
    """
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(initial_tempo=float(bpm) if bpm else 120.0)
    drums = pretty_midi.Instrument(program=0, is_drum=True)
    for rec in sorted(records, key=lambda r: r["onset"]):
        start = max(0.0, float(rec["onset"]))
        drums.notes.append(
            pretty_midi.Note(
                velocity=int(max(1, min(127, rec.get("velocity", 100)))),
                pitch=int(pitch_map[rec["class"]]),
                start=start,
                end=start + 0.05,
            )
        )
    midi.instruments.append(drums)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))
    return {
        "bpm": round(float(bpm), 4) if bpm else 120.0,
        "source": "argument" if bpm else "default_120",
        "notes": len(records),
    }


# --------------------------------------------------------------------------------------
# public core
# --------------------------------------------------------------------------------------


def _element_duration(signal: dict) -> float:
    return len(signal["band"]) / float(signal["sr"])


def _duration_mismatch_warning(signals: dict, audio_path: str | None, warnings: list[str]) -> dict:
    """Elements shorter than the audio silently kill every note past their end.

    A note at 200 s checked against a 90 s element finds no energy and looks
    "unsupported" - the fusion would drop real hits and report success. Say it out loud.
    """
    if not signals or not audio_path:
        return {}
    import soundfile as sf

    try:
        info = sf.info(str(audio_path))
        audio_duration = info.frames / float(info.samplerate)
    except Exception:
        return {}
    durations = {cls: round(_element_duration(sig), 2) for cls, sig in signals.items()}
    shortest = min(durations.values())
    if shortest + 0.5 < audio_duration:
        warnings.append(
            f"drum elements are shorter than the audio ({shortest:.1f} s vs {audio_duration:.1f} s): "
            "notes past the end of an element cannot be verified and will be dropped in "
            "strict/balanced/full modes - make sure the stems and the mix come from the same file"
        )
        logger.warning("element/audio duration mismatch: %.1f s vs %.1f s", shortest, audio_duration)
    return {"audio_duration_s": round(audio_duration, 2), "element_duration_s": durations}


def drums_midi_core(
    audio_path: str | None = None,
    output_dir: str = "drums_midi",
    element_stems_dir: str | None = None,
    extra_midi: list[str] | None = None,
    bpm: float | None = None,
    grid_origin_s: float | None = None,
    mode: str = "balanced",
    adt_of_thresholds: list[float] | None = None,
    allow_model_download: bool = True,
    quantize_to_grid: bool = True,
    max_shift_ms: float = 60.0,
    min_support_db: float = 1.5,
    lone_support_db: float = 3.0,
    split_by_class: bool = True,
    use_detector: bool = True,
    detector_classes: list[str] | None = None,
    model_classes: list[str] | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    """Build one drum MIDI (and per-class files) from audio and/or existing MIDI sources.

    Sources that vote: ADTOF on the mix, any `extra_midi` files, and (optionally) onset
    detection inside the separated drum elements. The elements are also the arbiter:
    a note must be louder than that element's own median level, unless nothing is
    provided to check against (then only the vote count applies and the report says so).

    `detector_classes` exists because the two source kinds cover different classes: a
    transcription model with a fixed vocabulary may have almost no toms or cymbals
    (measured: 16 and 6 over a whole track), while the detector is productive exactly
    there - and over-productive on toms, whose analysis band overlaps kick and snare.
    Restricting the detector to the classes the models miss is the honest middle.

    `model_classes` restricts the OTHER direction, and the kick is why it exists. On a
    neurofunk track the isolated kick element was measured to contain ~620 hits (a
    straight every-8th kick in places) while the model proposed 110 notes, of which 47
    had no hit within 30 ms of any element onset: the model both invented kicks and
    missed most real ones. A class whose physical source is available and clean should
    be taken from that source alone rather than arbitrated with a model that is wrong
    about it - pass model_classes=["snare", "hihat"] with
    detector_classes=["kick", "tom", "cymbal"] for exactly that split.
    """
    _configure_logging()

    if mode not in MODES:
        raise ValueError(f"mode must be one of: {', '.join(MODES)}")
    if audio_path is None and not extra_midi and not element_stems_dir:
        raise ValueError("at least one source is required: audio_path, extra_midi or element_stems_dir")
    # Validated EAGERLY, like detector_classes: a typo must fail on every path, not only on
    # the one where the model happens to run (with no audio_path the check below never fired,
    # so `model_classes=["kikc"]` was accepted in silence).
    if model_classes:
        unknown = sorted(set(model_classes) - set(CLASSES))
        if unknown:
            raise ValueError(f"model_classes has unknown classes: {unknown}; valid: {list(CLASSES)}")

    output_dir_p = Path(output_dir).expanduser()
    output_dir_p.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    notes_by_source: dict[str, list[tuple[str, float]]] = {}
    model_info: dict[str, Any] = {}

    if audio_path:
        audio = Path(audio_path).expanduser()
        _validate_input_file(audio)
        mapped, model_info = _notes_from_adt_of(
            audio,
            warnings,
            thresholds=adt_of_thresholds,
            device=device,
            allow_download=allow_model_download,
        )
        if mapped and model_classes:
            before = len(mapped)
            mapped = [item for item in mapped if item[0] in set(model_classes)]
            logger.info("model restricted to %s: %d of %d notes kept", model_classes, len(mapped), before)
        if mapped:
            notes_by_source["adt_of"] = mapped
        logger.info("adt_of: %d notes", len(mapped))

    for extra in extra_midi or []:
        extra_p = Path(extra).expanduser()
        if not extra_p.exists():
            warnings.append(f"extra_midi: not found: {extra_p}")
            continue
        label = f"midi:{extra_p.stem}"
        mapped = _notes_from_midi(extra_p, label, warnings)
        if mapped:
            notes_by_source[label] = mapped

    signals: dict[str, dict] = {}
    durations: dict[str, Any] = {}
    if element_stems_dir:
        element_dir = Path(element_stems_dir).expanduser()
        if not element_dir.is_dir():
            raise ValueError(f"element_stems_dir not found: {element_dir}")
        for cls in CLASSES:
            signal = _element_signal(element_dir, cls)
            if signal is None:
                warnings.append(f"no element found for '{cls}' in {element_dir.name}")
                continue
            signals[cls] = signal
        durations = _duration_mismatch_warning(signals, audio_path, warnings)
        if use_detector:
            detected = _notes_from_elements(signals, element_dir, warnings)
            if detector_classes:
                unknown = sorted(set(detector_classes) - set(CLASSES))
                if unknown:
                    raise ValueError(f"detector_classes has unknown classes: {unknown}; valid: {list(CLASSES)}")
                before = len(detected)
                detected = [item for item in detected if item[0] in set(detector_classes)]
                logger.info("detector restricted to %s: %d of %d onsets kept", detector_classes, len(detected), before)
            if detected:
                notes_by_source["detector"] = detected
            logger.info("detector: %d onsets", len(detected))

    if not notes_by_source:
        return {
            "ok": False,
            "reason": "no source produced any notes",
            "warnings": warnings,
            "sources": {},
        }

    # a fusion needs a step to cluster with; without bpm fall back to a 1/16 at 120 BPM
    cluster_step = (60.0 / bpm / 4) if bpm else 0.125
    clusters = _cluster(notes_by_source, cluster_step)
    kept = _select(clusters, signals, mode, min_support_db, lone_support_db)

    if quantize_to_grid:
        kept, grid_report = _quantize(kept, bpm, max_shift_ms / 1000.0, grid_origin_s)
    else:
        grid_report = {"applied": False, "reason": "quantize_to_grid=False"}

    _assign_velocity(kept, signals)

    pitch_map = dict(DEFAULT_OUTPUT_PITCH)
    combined_path = (output_dir_p / "drums.mid").resolve()
    tempo_info = _write_midi(kept, combined_path, pitch_map, bpm=bpm)
    if not bpm:
        warnings.append(
            "no bpm given: the MIDI files carry 120 bpm, so the item's length in a project "
            "of another tempo will not match the audio (pass the measured bpm)"
        )

    per_class: dict[str, str] = {}
    if split_by_class:
        for cls in CLASSES:
            class_records = [rec for rec in kept if rec["class"] == cls]
            if not class_records:
                continue
            class_path = (output_dir_p / f"drums_{cls}.mid").resolve()
            _write_midi(class_records, class_path, pitch_map, bpm=bpm)
            per_class[cls] = str(class_path)

    by_class: dict[str, int] = {cls: 0 for cls in CLASSES}
    for rec in kept:
        by_class[rec["class"]] += 1
    velocity_by_class = {
        cls: sorted({rec["velocity"] for rec in kept if rec["class"] == cls}) for cls in CLASSES
    }
    # Without elements there is nothing to MEASURE dynamics from, so every class ends up
    # with a single neutral value - the report must say that instead of implying dynamics.
    measured_classes = {cls for cls in CLASSES if cls in signals and VELOCITY_POLICY.get(cls, ("flat",))[0] == "measured"}

    return {
        "ok": True,
        "mode": mode,
        "audio": str(Path(audio_path).expanduser().resolve()) if audio_path else None,
        "output_dir": str(output_dir_p.resolve()),
        "midi": {"combined": str(combined_path), **({} if not split_by_class else {"by_class": per_class})},
        "midi_tempo": tempo_info,
        "notes": {"total": len(kept), "by_class": by_class, "clusters_considered": len(clusters)},
        "velocity": {cls: {"values": len(values), "range": [values[0], values[-1]] if values else None}
                     for cls, values in velocity_by_class.items()},
        "velocity_reference": {
            "source": "element_energy" if measured_classes else "none",
            "measured_classes": sorted(measured_classes),
            "note": (
                "dynamics measured from the drum elements"
                if measured_classes
                else "no elements given: every class gets one neutral value, NOT dynamics"
            ),
        },
        "grid": grid_report,
        "arbiter": {
            "elements_used": {cls: signal["files"] for cls, signal in signals.items()},
            "durations": durations,
            "min_support_db": min_support_db,
            "lone_support_db": lone_support_db,
        },
        "detector_classes": sorted(detector_classes) if detector_classes else "all",
        "model_classes": sorted(model_classes) if model_classes else "all",            # what the onsets were MEASURED in: a band whose hits do not stand above a busy
            # floor gives a count that is a property of the constants, not of the drum
            "detect_bands": {
                cls: {
                    "hz": signal["detect_band_hz"],
                    "refractory_ms": round(refractory_for_band(float(signal["detect_band_hz"][0])) * 1000.0, 1),
                    "contrast_db": signal["contrast_db"],
                    "stable": signal["contrast_db"] >= LOW_CONTRAST_DB,
                }
                for cls, signal in sorted(signals.items())
            },        "model": model_info,
        "sources": {label: len(notes) for label, notes in notes_by_source.items()},
        "warnings": warnings,
    }


def _configure_logging() -> None:
    from . import server  # local import: avoids a cycle at module load time

    server._configure_logging()


def _validate_input_file(path: Path) -> None:
    from . import server

    server._validate_input_file(path)
