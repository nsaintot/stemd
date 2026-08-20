# Model artefacts

This is what the hosting repository's README said before the source moved into
it. It describes the `models-v1` tag, which is superseded: the shipping
artefacts are safetensors under `models-v2`, and their digests are pinned in
`crates/stemd-server/src/models/preset.rs`.

It is kept because the provenance and licence position below apply to every
release, not only to the tag it was written for.

---

Hosting only. This repository carries the traced models that **stemd** downloads
on first run. The source lives elsewhere.

## `models-v1`

Two TorchScript artefacts traced for Apple MPS, each with its manifest:

| file | preset | size | sha256 |
|---|---|---|---|
| `htdemucs_mps.pt` | Quality | 161 MB | `9a5d56dac50cc58258df8576d8fa76f61d8943fbb117c43f55f06ce96c609d65` |
| `htdemucs_mps.json` | | 290 B | `f26265bc876887eac6934cc347a9a90c8da0b6b89d96199410180fed8c0564cc` |
| `hdemucs_mmi_mps.pt` | Speed | 320 MB | `dfed3230fb735772502bb4f9453810b84bf574a7d03ed6dc823b28b4f6327975` |
| `hdemucs_mmi_mps.json` | | 296 B | `00b131cc07b352f977c005485b991502d13f8c05a44813c3b3fe6169e2a18688` |

stemd pins these digests and verifies them before putting a download in place,
so replacing an asset here will make existing installs fail loudly rather than
quietly load something else. Publish a new tag instead of editing this one.

The artefacts are traced for MPS and will not load on CPU.

## Provenance

Traced from [facebookresearch/demucs](https://github.com/facebookresearch/demucs)
pretrained models: `htdemucs` (v4) and `hdemucs_mmi`.

The demucs **code** is MIT. The pretrained **weights** are distributed
separately from that repository and carry no explicit licence: the question was
asked in [demucs#327](https://github.com/facebookresearch/demucs/issues/327) in
2022, went unanswered, and the repository is now archived. These artefacts are
derived works of those weights and are published on that understanding.
