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
