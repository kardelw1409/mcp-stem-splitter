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
