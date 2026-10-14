# Vendored: ADTOF-pytorch

This folder contains an unmodified copy of the inference code from
[`xavriley/ADTOF-pytorch`](https://github.com/xavriley/ADTOF-pytorch), which is a PyTorch
port of ADTOF by Zehren, Alunno and Bientinesi:

* M. Zehren, M. Alunno, P. Bientinesi, "ADTOF: A large dataset of non-synthetic music for
  automatic drum transcription", ISMIR 2021.
* M. Zehren, M. Alunno, P. Bientinesi, "High-Quality and Reproducible Automatic Drum
  Transcription from Crowdsourced Data", Signals 4(4), 2023.

Files copied verbatim (`__init__.py`, `audio.py`, `model.py`, `post_processing.py`) plus
the model weights `data/adtof_frame_rnn_pytorch_weights.pth`. Verified byte-identical to
the upstream checkout by SHA-256; the weights hash is pinned in
`mcp_stem_splitter/adt_of.py` and checked on every use.

Why vendored: so this MCP has no dependency on a path on a particular machine - a fresh
install works offline, on any computer, without cloning a second repository.

## Licence

The upstream port carries no licence file of its own; the original ADTOF repository is
licensed **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0)**, and the weights inherit that. Practical consequences:

* non-commercial use only,
* attribution required (the citation above),
* derivative works must keep the same licence.

If this pipeline is ever used commercially or redistributed, this model must be replaced
(the same applies to the UVR/MDX-based separation models used elsewhere in this server).
