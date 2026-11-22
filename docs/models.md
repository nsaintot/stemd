# Model artefacts

The presets download their weights from this project's releases. stemd pins
each file's SHA-256 in `crates/stemd-server/src/models/preset.rs` and checks it
before it puts a download in place. Replacing an asset makes existing installs
fail with a digest error, so publish a new tag instead.

| release | files | preset |
| --- | --- | --- |
| [models-v2](https://github.com/nsaintot/stemd/releases/tag/models-v2) | `htdemucs.safetensors`, `htdemucs_ft.safetensors` | Fast, Balanced |
| [models-v3](https://github.com/nsaintot/stemd/releases/tag/models-v3) | `stemd_roformer.safetensors` | Quality |

## `models-v2`

demucs v4 in safetensors. The files are byte-for-byte copies of
[mlx-community's](https://huggingface.co/mlx-community/demucs-mlx) conversions.

| file | preset | bytes | sha256 |
| --- | --- | --- | --- |
| `htdemucs.safetensors` | Fast | 168,005,865 | `339d267a7a6983a11eedbdc00413c602a65e9b9103f695fb5c2b2a481cd9d297` |
| `htdemucs_ft.safetensors` | Balanced | 672,024,519 | `53f03b1ad4b4d211025a35da65460ba61a17547adf9c0544cad0ebcc8d7bbabb` |

### Provenance

The conversions come from the
[facebookresearch/demucs](https://github.com/facebookresearch/demucs) pretrained
models `htdemucs` and `htdemucs_ft`.

The demucs code is MIT. The pretrained weights are distributed separately from
that repository and carry no explicit licence. The question was asked in
[demucs#327](https://github.com/facebookresearch/demucs/issues/327) in 2022 and
went unanswered, and the repository is now archived. These artefacts are
derived works of those weights and are published on that understanding.

## `models-v3`

`stemd_roformer`, stemd's own separation model, which the `Quality` preset runs.

| | |
|---|---|
| File | `stemd_roformer.safetensors`, 250,585,752 bytes |
| SHA-256 | `ce3802b634d04d955626cffe038ec5ac8f26ef1c90f3d4cfe5ac542ad6da4b37` |
| Architecture | BS-RoFormer: dim 256, depth 12, 62 bands, rotary position, 125.2M parameters, float16 |
| Output | bass, drums, other, vocals; stemd ships drums and vocals and builds harmonics as the remainder |
| Release | [models-v3](https://github.com/nsaintot/stemd/releases/tag/models-v3) |

### Terms

Copyright 2026 Nicolas Saintot.

The weights in `stemd_roformer.safetensors` are licensed under the
[Creative Commons Attribution-NonCommercial 4.0 International
licence](https://creativecommons.org/licenses/by-nc/4.0/legalcode) (CC BY-NC
4.0). In short:

- You may copy, redistribute, convert and fine-tune the weights.
- Credit stemd, link to this licence, and say if you changed them.
- You may not use the weights, or a model derived from them, for commercial
  purposes. Selling them, bundling them in a paid product, or offering a paid
  service built on them needs written permission.

These terms cover the model only. They place no restriction on the audio
produced with it: stems separated by stemd, or by any other software running
these weights, belong to whoever holds the rights in the source recording, and
may be used however those rights allow.

The licence applies to these weights alone. stemd's code is MIT or Apache-2.0,
at your choice; the other models stemd downloads carry their own terms, listed
above.

Training data includes MUSDB18-HQ (train split), MoisesDB, Slakh2100 and CC0
recordings. Some of these datasets allow non-commercial research use only,
which is why the weights are non-commercial. No MUSDB18-HQ test track was used
for training.

The weights are provided as is, without warranty of any kind, and the licensor
is not liable for any damages arising from their use, as set out in sections 5
and 6 of the licence.

To ask about commercial use, open an issue on
[nsaintot/stemd](https://github.com/nsaintot/stemd/issues).

## Retired

`models-v1` holds two TorchScript artefacts traced for Apple MPS, each with
its manifest. Nothing loads TorchScript any more; a settings file that names one
falls back to the default preset with an explanation.

| file | former preset | size | sha256 |
|---|---|---|---|
| `htdemucs_mps.pt` | Quality | 161 MB | `9a5d56dac50cc58258df8576d8fa76f61d8943fbb117c43f55f06ce96c609d65` |
| `htdemucs_mps.json` | | 290 B | `f26265bc876887eac6934cc347a9a90c8da0b6b89d96199410180fed8c0564cc` |
| `hdemucs_mmi_mps.pt` | Speed | 320 MB | `dfed3230fb735772502bb4f9453810b84bf574a7d03ed6dc823b28b4f6327975` |
| `hdemucs_mmi_mps.json` | | 296 B | `00b131cc07b352f977c005485b991502d13f8c05a44813c3b3fe6169e2a18688` |

They were traced from the demucs pretrained models `htdemucs` (v4) and
`hdemucs_mmi`, under the same terms as `models-v2`.
