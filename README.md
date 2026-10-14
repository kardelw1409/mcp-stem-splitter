# mcp-stem-splitter

MCP server (`stdio`) that splits an audio file into 4 stems (`drums/bass/vocals/other`) and returns absolute paths.

Run:

```powershell
py -3.11 -m mcp_stem_splitter
```

Notes:
- Default output names are fixed: `drums.wav`, `bass.wav`, `vocals.wav`, `other.wav`.
- To prefix with the input name, call `split_stems(..., filename_mode="prefixed")` which produces e.g. `zgaryshcha_bass.wav`.
- GPU: by default `device="auto"` uses CUDA if your installed PyTorch build supports it; otherwise it will fall back to CPU. You can force `device="cuda"` (or `cuda:0`) or try `device="directml"` if you install `torch-directml`.

Tools:
- `list_models()` -> `{ models, presets, notes }` (curated list)
- `get_presets()` -> `{ presets: [...] }`
- `split_stems(input_path, output_dir, ...)` -> `{ stems: { drums,bass,vocals,other } }`
- `split_vocals_only(input_path, output_dir, ...)` -> `{ stems: { vocals,instrumental } }`
- `drums_midi(...)` -> drum MIDI, one file per drum class (see below)

## Drums MIDI (`drums_midi`)

Builds a drum MIDI part for a track, optionally split into one file per drum class so a
DAW can place kick / snare / hi-hats / toms / cymbals as **separate items**.

```python
drums_midi(
    audio_path=r"...\mix.wav",              # a transcription source (ADTOF)
    element_stems_dir=r"...\drumsep_out",   # kick/snare/hh/toms/ride/crash stems
    extra_midi=[r"...\adtstr.mid"],         # any number of extra sources
    output_dir=r"...\drums_midi",
    bpm=172.0,                              # measured tempo; omit to skip quantisation
    grid_origin_s=0.031,                    # measured grid phase (from the manifest)
    mode="balanced",
)
```

Returns `{ midi: { combined, by_class: {kick, snare, hihat, tom, cymbal} }, notes, velocity, grid, arbiter, sources, warnings }`.

Recommended call (the hybrid measured on a 270 s drum & bass track):

```python
drums_midi(
    audio_path=r"...\mix.wav",
    element_stems_dir=r"...\drumsep_out",
    output_dir=r"...\drums_midi",
    bpm=172.0135,                     # MEASURED tempo
    grid_origin_s=0.1595,             # MEASURED phase, e.g. from the audiocpp manifest
    detector_classes=["tom", "cymbal"],   # see below
)
```

Why restrict the detector: a 5-class transcription model produced 110 kicks / 227 snares /
730 hi-hats but only **16 toms and 6 cymbals** over the whole track (its vocabulary has no
ride/crash), while onset detection inside the elements is productive exactly there. The
same run with an unrestricted detector gives 4140 notes (~22 per bar - more than a drummer
plays); with none, 1089 and no toms/cymbals. The hybrid gives 2796.

What to read in the result before calling it a success:

| field | meaning |
| --- | --- |
| `warnings` | a missing element, an unreadable source, or **elements shorter than the audio** (then every note past their end is dropped - stems and mix are not from the same files) |
| `grid.origin_source` | `explicit` when you passed `grid_origin_s`, `estimated_from_notes` otherwise (an estimate cannot fix a systematic offset) |
| `velocity_reference.source` | `none` means no elements were given: every class got one neutral value, **not** dynamics |
| `notes.by_class` | where the notes came from - use it to spot a class that a source cannot produce at all |

Why more than one model:

* a single model's notes are only about half supported by the audio (measured),
* even four models agreeing leave ~a quarter of notes unsupported - they share a bias,
* the separated drum elements are the only physical ground truth available, and they
  are independent of every model.

So every note must be **proposed by a source** and **supported by the matching element**
(louder than that element's own median level - a self-calibrating test, no invented
threshold). Modes trade precision for recall:

| mode | keeps |
| --- | --- |
| `consensus` | any 2+ sources agree (no physical check) |
| `strict` | 2+ agree **and** the element supports it |
| `balanced` (default) | as strict, or a lone source that is clearly above the element level |
| `full` | anything the element supports (recall first) |

Velocity is set **per class**, because the models are not equally good at it: kick is
flat (its model velocity correlates +0.09 with the audio - a kick is a sample trigger),
snare/hi-hat get measured dynamics (correlations +0.65 / +0.45), toms and cymbals a
compressed range.

`element_stems_dir` must contain the separated elements (e.g. `kick.wav`, `snare.wav`,
`hh.wav`, `toms.wav`, `ride.wav`, `crash.wav` - the `drumsep_(kick)_....wav` naming is
also recognised). Without it, notes fall back to source agreement only and the report
says `support_db: null` rather than pretending they were verified.

## Model assets (nothing to install, no configured paths)

The transcription model (`mcp_stem_splitter/vendor/adtof_pytorch`, a port of Zehren et
al., see `NOTICE.md`) and its weights (`data/adtof_frame_rnn_pytorch_weights.pth`,
3.6 MB) ship **inside this package**, so the server works on a fresh machine with no
clone, no `PYTHONPATH` and no hardcoded path. Resolution order, first hit wins:

1. `MCP_STEM_SPLITTER_ADTOF_WEIGHTS` - explicit file, an error if it is missing (a
   silent fallback would hide a typo),
2. the bundled file,
3. a cache directory (`MCP_STEM_SPLITTER_CACHE`, else `%LOCALAPPDATA%\mcp-stem-splitter`
   on Windows, else `~/.cache/mcp-stem-splitter`),
4. a one-time download from the upstream repository - only when
   `allow_model_download=True` (the default; pass `False` for strictly offline runs).

The file is checked against a pinned SHA-256 **before** it is loaded, so a truncated or
corrupted blob is refused instead of silently running with random weights (upstream
loads weights only `if the path exists`, and a random-weight model still emits plausible
MIDI - that is why the digest is pinned rather than trusted). Upstream prints to stdout,
which is the JSON-RPC channel, so the library's output is redirected to stderr.

The result's `model` field records the exact file, where it came from (`vendored`,
`env`, `cache`, `downloaded`), its digest, the `device` used and the thresholds in
force - report those instead of assuming the model loaded.

## Installing on another machine

Nothing in this repository points at a path on the machine it was written on. A fresh copy is
self-sufficient for `drums_midi`:

```powershell
git clone https://github.com/kardelw1409/mcp-stem-splitter.git
cd mcp-stem-splitter
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
```

For GPU, install a CUDA build of PyTorch **before** that last line, so pip does not resolve a CPU
one first (`device="auto"` then picks CUDA, and the result's `model.device` says which was used):

```powershell
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Then point the MCP entry at that venv's `python.exe` - in the *consuming* workspace's
`.vscode/mcp.json` (this repository has no `.vscode/`, it is git-ignored):

```json
{
  "servers": {
    "stem_splitter": {
      "type": "stdio",
      "command": "C:\\path\\to\\mcp-stem-splitter\\.venv\\Scripts\\python.exe",
      "cwd": "C:\\path\\to\\mcp-stem-splitter",
      "args": ["-m", "mcp_stem_splitter"]
    }
  }
}
```

An already-running MCP server keeps the tool list it started with, so after changing tool
descriptions or adding a tool, restart the server (or reload the window) before expecting to see it.

### What comes with the clone, and what fetches itself

| asset | size | how it arrives |
| --- | --- | --- |
| ADTOF code + weights (`drums_midi`) | 3.5 MB | **inside the package** - already in the clone |
| Demucs weights (`split_stems`) | 48-160 MB per model | Demucs downloads them on first use into `%USERPROFILE%\.cache\torch\hub\checkpoints` |
| DrumSep checkpoint (the `element_stems_dir` source) | 417 MB | downloaded by `audio-separator`; needed only for per-element arbitration |
| ADTOF weights, if a slim install dropped them | 3.5 MB | one fetch into `%LOCALAPPDATA%\mcp-stem-splitter`, reused after that; `allow_model_download=False` refuses it |

A clone of the whole repository is small: measured at `.git` 1.94 MB plus a 3.56 MB working tree
(18 files), with no virtualenv and no model caches in it - those are machine-local and are built by
`pip` and by the tools themselves.

`requirements.lock.txt` is the exact `pip freeze` of the venv these numbers were measured in, for
when a bug has to be reproduced rather than a tool used. Three things in it are worth knowing
before copying it verbatim:

* `mcp<2` is a **hard requirement**, not tidiness. mcp 2.x renamed `FastMCP` to `MCPServer` and
  changed the decorator/run API, so `mcp>=1.0.0` resolved 2.3.0 on a clean venv and the server
  could not even be imported (`No module named 'mcp.server.fastmcp'`). `pyproject.toml` pins it.
* its `torch` is a **nightly** (`2.11.0.dev20260118+cu128`) - nightly builds are pruned from the
  index, so that exact pin stops being installable; use a current nightly or a stable release,
* it also contains packages that belong to other experiments and are not needed here:
  `tensorflow` / `tensorflow-intel` (pulled in by `basic-pitch`), `torchaudio`, `torchcodec`.
  This matters, because **tensorflow is what holds `numpy` at 1.23.5 and `scipy` at 1.10.1** in
  that venv - `onnx` once pulled numpy 2 and broke both (`_ARRAY_API not found`).

The two `-e git+...` entries that a `pip freeze` of that venv does list are deliberately excluded:
the project itself (install it from your checkout) and the old ADTOF clone, which is obsolete now
that its four modules and its weights ship inside the package.

This is not theoretical - it was measured on a clean venv built exactly as above (Python 3.11.9,
`pip install .`, no other packages): pip resolved `mcp 2.3.0` and the server did not import (that is
what the pin fixes); with the pin it resolved `mcp 1.30.0`, listed all five tools, passed 32/32 of
the package's own tests and produced a real transcription of the 90 s drum & bass excerpt - 707
notes, `weights_source: vendored`, `device: cpu` (auto-selected, since that torch was CPU-only),
`numpy 2.4.6`, `scipy 1.17.1`, `torch 2.14.1+cpu`. So the tools run on a modern stack; the old
pins in `requirements.lock.txt` are the constraint of the machine this venv lives on, not of the
code.
