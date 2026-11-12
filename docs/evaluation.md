# Evaluation

Measurements behind the presets. SDR is global SDR (MDX/multisong convention),
in dB, higher is better. Each timing names its machine: an M1 Pro (MLX on Metal)
or an RTX 3090 Ti (MLX on CUDA).

## Every preset, per part

Global SDR over the first 12 MUSDB18-HQ test tracks, full length, through
`stemd-server` and `stemd-cli` 0.1.4 on the RTX 3090 Ti. The separation code is
unchanged in 0.2.0.

| preset | drums | bass | other | harmonics | vocals | realtime, 3090 Ti |
|---|---|---|---|---|---|---|
| Fast | 8.27 | 8.17 | 5.14 | 8.62 | 8.92 | 99x |
| Balanced | 8.49 | **8.40** | 5.20 | 8.78 | 9.28 | 56x |
| Quality | **11.07** | 7.97 | **6.52** | **11.54** | **11.34** | 18x |

- `drums`, `harmonics` and `vocals` come from the default request, with `drums`
  rebuilt from the mix. `bass` and `other` come from `parts=bass,other,vocals`.
- Realtime is the mean over the full tracks, cold start included.
- Some tracks pull every preset down. On "AM Contra - Heart Peripheral" all
  three put synths in `bass` (-1.33 to -0.68 dB). On "Arise - Run Run Run"
  stemd's htdemucs scores `other` at 0.88 dB, the same as the reference
  PyTorch htdemucs.

## Test sets

| set | used for | limits |
|---|---|---|
| 12 MUSDB18-HQ test tracks, full length | the presets as shipped | `stemd_roformer` was not trained on them |
| 25 MUSDB clips of 6.8 s | model and arrangement choices before `stemd_roformer` | include MUSDB train tracks; one padded segment per clip, so blind to segment boundaries and `--overlap`; differences under about 0.2 dB are noise |
| 45 s excerpts of electronic tracks | leakage between stems, listening | no ground truth |

MUSDB is mostly rock and pop. The target material is electronic, and the two
disagree where it matters most: processed vocals and tracks without vocals.

## Presets

| preset | runs | why |
|---|---|---|
| Fast (default) | `htdemucs` v4, one pass, `drums` rebuilt | won the listening test; 18x realtime on the M1 Pro |
| Balanced | `htdemucs_ft`'s drums and vocals models, `harmonics` as the remainder | two of the bag's four models: 9.83 mean on the clips against 9.47 for all four, and 14.5 s against 25.8 s on a 120 s track (M1 Pro) |
| Quality | `stemd_roformer`, one pass, `harmonics` as the remainder | against the BS PolarFormer and `htdemucs_ft` chain it replaced: drums +2.3, harmonics +1.7, vocals -1.2 dB, 1.6x to 1.8x faster, one 251 MB artefact instead of 774 MB |

`stemd_roformer` is a BS-RoFormer trained for stemd: PolarFormer's shape (dim
256, hop 512, chunk 588,800) with rotary position and four mask estimators,
125.2M parameters. The MLX port nulls against the PyTorch reference at -117 dB
on all four stems. Metal runs it at float16 and CUDA at float32; SDR is the same
to the second decimal.

`Balanced` runs a third model, the bag's bass specialist, only when a job asks
for `bass` or `other`.

## Retired and rejected

| model | result | why not |
|---|---|---|
| `hdemucs_mmi` (v3) | 9.43 on the clips, the highest | its LSTM runs 3.8x slower on MLX than on torch; keeping it meant keeping libtorch |
| BS-RoFormer viperx, vocals for Quality | vocals 11.12 on the clips | leaves voice in `harmonics` on electronic material: -39.0 dB of the other model's vocals against -52.5 dB for PolarFormer |
| BS PolarFormer and `htdemucs_ft`, chained | 10.39 mean on the 12 tracks | replaced by `stemd_roformer` (11.32) |
| MelBand RoFormer | vocals +2.09 over `hdemucs_mmi` | one stem only; no 4-stem checkpoint found |
| Open-Unmix UMX-L | distorted vocals, percussion in bass | quality |
| `mdx_extra_q` | | 201 s for a 6:38 track |
| SCNet Large | | 0.51x realtime, LSTM-heavy, no GPU path |

## Arrangement

When two models make the stems, `harmonics` ships as the remainder,
`mix - vocals - drums`, so the rebuilt `drums` is the drums model's own output.

| config, 25 clips | drums | harmonics | vocals | mean |
|---|---|---|---|---|
| `htdemucs` (Fast) | 8.04 | 10.34 | 8.72 | 9.03 |
| `htdemucs`, `harmonics` as the remainder | 8.10 | 10.33 | 8.72 | 9.05 |
| RoFormer vocals, `htdemucs` raw `bass + other` | 5.70 | 10.34 | 11.12 | 9.05 |
| RoFormer vocals, `htdemucs` drums, `harmonics` as the remainder | 8.10 | 11.32 | 11.12 | 10.18 |
| RoFormer vocals, `htdemucs_ft` drums, `harmonics` as the remainder | 9.10 | 11.96 | 11.12 | 10.73 |
| `htdemucs_ft`, all four models | 8.27 | 10.88 | 9.26 | 9.47 |
| `htdemucs_ft`, drums and vocals, `harmonics` as the remainder (Balanced) | 9.10 | 11.12 | 9.26 | 9.83 |

- Raw `bass + other` from one model with vocals from another puts the models'
  disagreement about the vocals into the rebuilt drums.
- Fast stays one raw pass: the remainder changes nothing for a single model
  whose sources sum to -33.9 dB of the mix.
- Feeding the drums model `mix - vocals` (chaining) helps only behind a
  RoFormer: vocal content shared with drums went from -29.6 to -34.2 dB. Behind
  Balanced's demucs vocals it got worse, -28.2 to -27.3 dB, so Balanced does
  not chain.

## Which part to rebuild

The rebuilt part carries the model's whole residual. SDR cannot choose: the
three choices land within 0.04 dB. On 90 s of real material with `htdemucs`, the
residual is -29.8 dB of the mix with a crest factor of 14.3, so it sounds
percussive.

| rebuilt | its level | residual below it | drums fader down |
|---|---|---|---|
| **drums** | -3.7 dB | 26.0 dB | clean |
| harmonics | -2.6 dB | 27.2 dB | ghost hits at -27.3 dB RMS |
| vocals | -31.3 dB | -2.6 dB | ghost hits at -27.3 dB RMS |

- Level: the vocals stem can sit below the residual on instrumental material.
- Placement: only rebuilding `drums` lets the drums fader remove the
  percussive residual with the drums.

## Listening

Electronic material, with the arrangements above. Quality was then the viperx
chain.

| preset | heard | vocals content shared with drums |
|---|---|---|
| Fast | compression and sidechain artefacts, tick residues | -25.3 dB |
| Balanced | the same, less compression | -28.2 dB |
| Quality | clean vocal isolation | -34.2 dB |

On a track without vocals, `htdemucs_ft`'s vocals model puts a signal peaking
at 0.499 in the vocals stem; a BS-RoFormer returns 1e-4.

## Runtime

| measurement | result |
|---|---|
| MLX port of `htdemucs` against torch/MPS, 120 s track, M1 Pro | 6.5 s against 14.7 s (2.3x) |
| MLX against TorchScript, real music | stems null at -54.0 dB; residual identical (-33.9 dB) |
| float16 | 1.3x faster; -53.9 dB for `htdemucs`, -54.8 dB for RoFormer vocals |
| first separation after launch | about twice the warm time while kernels compile |
| `htdemucs` segment 10 s instead of its native 7.8 s | 5% slower; stemd runs every model at its native segment |
| full server round trip against inference alone (`hdemucs_mmi`) | 21.6x against 44.1x realtime: single-threaded DSP around the model |

## Tuning levers

| lever | quality | cost | verdict |
|---|---|---|---|
| overlap 0.50 | about +0.03 dB | 30 s | below noise |
| shifts=1 / shifts=2 | -0.07 / -0.02 dB | 2x | negative |
| ensemble with `htdemucs` | +0.11 dB | 51 s | over budget |
| which part is rebuilt | 0.04 dB | | noise |
| soft-mask refinement | **-1.55 dB** | | the mix's phase replaces demucs's |
| shipping all three parts instead of rebuilding one | 9.48 against 9.44 to 9.48 | | exact reconstruction is nearly free |

Two independently trained models disagree by -22 dB of the mix on drums and
harmonics. That misassignment is the dominant error, and every lever above
sits 14 dB or more below it.

## On-device separation

Why separation runs on a server: an RK3399 client (4x Cortex-A53, 2x
Cortex-A72) reaches 7.52 GFLOP/s fp32 on its 5 usable cores, against 50.45 on
one M1 core. Hybrid Demucs v3 needs 2.31 s of compute per second of audio there.
Removing its LSTM and time-domain branch together buys 1.52x of the 5x needed.

## Reproduce

| what | how |
|---|---|
| per-part table | `tools/eval/presets.py` |
| 25-clip benchmark | `uv run --extra eval tools/eval/benchmark.py` |
| `stemd_roformer` against the chain | `stemd_roformer_against_quality` in `crates/stemd-core/tests/separating.rs` |
| arrangements | `tools/eval/roformer_hybrid.py` |
| PolarFormer quality and speed | `tools/eval/polarformer.py`, `tools/eval/roformer_speed.py` |
| leakage on excerpts | `what_each_preset_leaves_in_the_vocals`, `chaining_the_halves_against_running_them_side_by_side` |
| weights | `tools/export/convert_stemd_roformer.py`, then `tools/export/convert_roformer.py` |
