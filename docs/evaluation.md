# Evaluation record

Measurements behind the configuration in the README. M1 Pro, 44.1 kHz stereo,
MLX on Metal unless a row says otherwise: every timing and every null below is
that machine and that backend, and neither transfers. What transfers is the
method and, usually, the ratio between two rows measured the same way.
SDR is global SDR (MDX/multisong convention) over 25 MUSDB clips, scored on the
three parts a player ends up with.

Reproduce any row with `uv run --extra eval tools/eval/benchmark.py`.

## Models

| candidate | quality | cost (6:38 track) | status |
|---|---|---|---|
| **htdemucs** v4 | 8.97 dB SDR | 43.9 s | **Fast preset, the default**, and what won by ear |
| htdemucs_ft | ties hdemucs_mmi (9.15 vs 9.19) | 174 s | Balanced preset |
| hdemucs_mmi | **9.43 dB SDR** | **19.2 s** | **retired**: see MLX, below |
| Open-Unmix (UMX-L) | distorted vocals, percussion in bass | fast | unusable |
| mdx_extra_q |: | 201 s | bag of 4, weights not identity |
| SCNet Large |: | 0.51x realtime | 59% LSTM, no GPU path |
| MelBand RoFormer | **+2.09 dB vocals, +1.96 dB instrumental** | ~20 min/track on torch, **1.6 on MLX** | vocals only: see below |

The presets are a cost ladder, not a quality ladder: the cheapest scores highest
here and the most expensive does not win. The table is a strong guide to cost and
a weak guide to quality: MUSDB is largely rock and pop, the target material is
electronic, model performance is genre-dependent, and SDR penalises
waveform-level deviation that is perceptually harmless.

`hdemucs_mmi` scores highest here and is no longer offered, which needs saying
plainly: it was retired for a reason that has nothing to do with quality. It is
v3, the one architecture MLX cannot run well, and keeping it would have meant
keeping libtorch to serve a single preset: see MLX below. `htdemucs` is the
default in its place, which the listening test already preferred: clearly better
on harmonics and drums, indistinguishable on vocals. The benchmark and the ears
disagreed, and the runtime settled it.

### MelBand RoFormer, in detail

Against MUSDB ground truth over 12 clips:

| | drums | bass | other | vocals | instrumental |
|---|---|---|---|---|---|
| hdemucs_mmi | 8.81 | 10.94 | 6.60 | 9.00 | 14.53 |
| MelBand RoFormer | 9.97 | 11.08 | 8.24 | 11.08 | 16.50 |
| difference | +1.16 | +0.14 | +1.64 | **+2.09** | **+1.96** |

It separates better on every source and leaves *more* energy unassigned:
sum-residual −30.7 dB against hdemucs_mmi's −35.6 dB. Both are true. Neither
model is trained with any constraint that its sources sum to the input, so a
better separator is under no obligation to leave a smaller leftover. The residual
governs how much junk rides the rebuilt fader, and nothing else.

MPS runs it **slower than CPU**: 0.38x realtime against 0.50x. The blocking
error (`scatter(): Yet not supported for complex`) is fixable in ten lines:
`view_as_real`, `scatter_add_`, `view_as_complex`, verified numerically identical
at a −83.4 dB null, and fixing it does not help. The model is many small
transformer ops on complex tensors that fall back across the device boundary, and
the transfers cost more than the GPU saves.

**MLX removes that boundary, and with it the verdict.** The same architecture
through `mlx_audio.sts.models.mel_roformer` runs at **4.36x realtime**: against
torch/MPS's 0.38x, an **11x swing**, and the same cost tier as `htdemucs_ft` on
MLX (4.32x). A model filed here as twenty minutes a track is about ninety seconds
of one. The diagnosis above was right and it was a diagnosis of *torch*, not of
the architecture.

Two caveats on that figure. It was taken over ten-second chunks with hard
boundaries and no overlap-add, so it is an upper bound on speed: real inference
overlaps its windows and would roughly double the cost, to about 2.2x. And it is
one checkpoint on synthetic material.

**What blocks it is the stem count, not the speed.** This server sells three
controls, and needs `harmonics` and `vocals` to get them. Every RoFormer that can
be run here today emits fewer:

| checkpoint | stems |
|---|---|
| `mel-roformer-kim-vocal-2-mlx` | **1**: `vocals`, instrumental by subtraction |
| BS-Roformer-Viperx-1297 | 2: vocals, instrumental |

Neither splits `drums` from `harmonics`, which is the split this exists to make.
Pairing one with demucs for the rest would also cost exact reconstruction: parts
from two different models have no reason to sum back to the mix.

There is no BS-RoFormer on MLX at all: `mlx-audio` implements Mel-Band only,
and its four presets are all `num_stems=1` despite one being named
`zfturbo_bs_roformer`.

**The way in, if it is worth taking:** `MelRoFormerConfig.num_stems` is a real
field and `convert_checkpoint` takes arbitrary PyTorch checkpoints, so a 4-stem
Mel-Band RoFormer would convert into an implementation that already runs at this
speed. That lands a model scoring 2 dB above `hdemucs_mmi` at roughly
`htdemucs_ft`'s cost. Finding such a checkpoint is the open step; none of the
above is worth acting on until one exists and has been converted and nulled.

## MLX

[mlx-community/demucs-mlx](https://huggingface.co/mlx-community/demucs-mlx)
publishes converted weights for every preset, so the question is whether Apple's
own array framework beats torch/MPS on the same GPU. Measured against this
server on the same machine and the same audio, matched settings: one pass,
overlap 0.25, the model's native segment:

| model | torch/MPS | MLX | |
|---|---|---|---|
| htdemucs, 2:00 | 14.7 s (8.2x) | **6.8 s (17.6x)** | MLX 2.2x faster |
| htdemucs, 7:00 | 47.6 s (8.8x) | **26.2 s (16.0x)** | MLX 1.8x faster |
| htdemucs_ft, 2:00 | 55.6 s (2.2x) | **27.8 s (4.3x)** | MLX 2.0x faster |
| hdemucs_mmi, 2:00 | **7.7 s (15.6x)** | 29.3 s (4.1x) | MLX 3.8x *slower* |

**The split is architectural, and it is the wrong way round for us.** htdemucs v4
is convolution plus a cross-domain transformer: matmul and attention, which is
what MLX is built for, and its 7.8 s segment means many small forward passes,
where lazy evaluation and kernel fusion earn their keep. hdemucs_mmi is v3:
convolution plus a bidirectional LSTM, a long sequential chain of small matmuls
with no fused kernel behind it, run over a 44 s segment. It is the same shape of
problem that made an Open-Unmix port unattractive years earlier.

So MLX would roughly halve the cost of Balanced and quadruple the cost of Speed,
which is the default and the cheapest thing here. There is no configuration that
takes only the win: serving both would mean keeping libtorch as well, and the
strongest argument for MLX was never the speed, it was deleting libtorch, the
223 MB of dylibs, `build-env.sh`, the venv, and the `_exit` teardown hack.

**The outputs also disagree**, which matters more than the timings. Nulling MLX
against this server for the same input, on `harmonics`, which is raw model
output on both sides:

| model | difference, relative to the stem |
|---|---|
| htdemucs | −36 dB |
| hdemucs_mmi | −22 dB, and its `vocals` runs 15 dB hotter |

−36 dB is below this model's own residual and probably inaudible; −22 dB is not
obviously either. Neither is float32 rounding, so the two runtimes are not
computing the same thing, and nothing here says which is right. The conversion
was not verified against the original checkpoints, that would want the torch
weights alongside, so the divergence could as easily be the conversion as the
runtime. On synthetic material, at that.

### Dropping v3: done

Every model that gains is v4, so `hdemucs_mmi` was retired and the ladder moved
up: `htdemucs` is Fast and the default, `htdemucs_ft` is Balanced. One runtime
can now serve the whole menu, which is what makes the MLX switch possible at all.

The cheap tier ends up *faster* than it was, but only after that switch: on
torch, `htdemucs` runs at 8.2x against the retired preset's 15.6x, so until MLX
lands this is a 2x regression at the cheap end, knowingly taken. On MLX it
becomes 17.6x, ahead of where it started, on the model that won the listening
test. `htdemucs_ft` halves.

Two things still to settle.

**The runtimes agree on htdemucs and not on the bag.** Residual against the mix,
and the null between the two implementations' `harmonics`:

| model | torch | MLX | they differ by | `harmonics` null |
|---|---|---|---|---|
| htdemucs | −33.9 dB | −33.8 dB | **0.1 dB** | −36 dB |
| hdemucs_mmi | −27.2 dB | −28.3 dB | 1.1 dB | −22 dB |
| htdemucs_ft | −13.9 dB | −20.0 dB | **6.1 dB** | −14 dB |

htdemucs is the one the two runtimes plainly agree on. The bag is the one they do
not, by enough to matter, and this does not say which is right: it could be the
traced artefact, the MLX conversion, or a synthetic test signal flattering
neither. Resolving it wants the official checkpoints on both sides and real
material.

**And `htdemucs_ft` leaves 20 dB more unaccounted for than `htdemucs`: on both
runtimes.** That is not a bug and not about MLX: with identity bag weights each
source comes from a *different* fine-tuned model, so the four have no reason to
sum to the mix. Here it costs the rebuilt part, which carries the whole residual,
so the drums control on Extreme rides roughly −14 dB of junk against htdemucs's
−34 dB. Already true of Extreme today, but it stops being a corner case if the
bag becomes the quality tier.

**Not pursued** on the strength of these numbers alone: measured on synthetic
material, which is the wrong evidence for a decision about how something sounds.

### The Rust port: done, and it settles the above

MLX in Rust rather than through the Python package, so the venv and libtorch
could both go. Every stage nulled against the reference implementation as it was
built: spectrogram −139 dB, encoder and decoder −120 to −143 dB, the cross-domain
transformer −107 to −150 dB, a whole segment −124.8 dB, a whole track −123.7 dB.

That resolves the open question above. The two runtimes were never far apart on
real material: the −36 dB was synthetic. Against the TorchScript backend on a
120 s track of actual music, the shipped stems null at **−54.0 dB** with the
residual identical to a tenth of a dB (−33.9 both). A runtime swap 20 dB below
the model's own error is a runtime swap.

| | 120 s track | realtime | vs torch |
|---|---|---|---|
| torch/MPS | 14.7 s | 8.2x |: |
| MLX, first working port | 15.5 s | 7.8x | *slower* |
| + inverse transform fixed | 9.5 s | 12.7x | 1.5x |
| + float16 | 7.8 s | 15.3x | 1.9x |
| **+ fused normalisation** | **6.5 s** | **18.4x** | **2.3x** |
| htdemucs_ft, same | 25.8 s | 4.6x | 2.2x |

The projection at the top of this section said 17.6x for htdemucs and 4.3x for
the bag. Both were beaten, and the finished port is also faster than the Python
it was nulled against: 6.5 s against 6.65 s with that package's hand-written
Metal kernels, 7.02 s without.

The last row is both runtimes running `htdemucs_ft` the way it was run then: all
four models, combined by an identity matrix. It is what the runtime swap was
worth on that workload, not what Balanced costs today; that is 14.5 s, and
[further down](#the-same-arrangement-applied-to-balanced) is why.

Worth recording *how*, because neither win was clever. The first port was slower
than torch, and the two fixes were both cases of writing out by hand something
MLX already had: an inverse transform that walked 336 frames instead of doing
four reshapes (40% of a forward pass), and normalisations written as eleven
tensor passes where `fast::layer_norm` is one fused kernel, sixty-four times per
forward. Half precision is worth 1.3x and costs −53.9 dB, the same distance as
the runtime change, with the residual again unmoved.

The bag disagreement in the table above was the traced artefact, not the
conversion: with both runtimes given the same official weights, `htdemucs_ft`
reports −13.9 dB on MLX, matching torch exactly.

Four of the five things listed above as the real argument for MLX did go: the
dylibs, `build-env.sh`, the venv, the pinned torch. The `_exit` teardown stayed.
MLX has not been seen to abort on the way out the way libtorch's MPS allocator
did, but it is the same shape of hazard, and twenty lines is cheaper than finding
out on someone else's machine.

## A quality tier: BS-RoFormer for the vocals

Measured before porting anything, because the port is the expensive part and
this is what says whether it is worth doing. Same 25 MUSDB clips, same global
SDR, as everything else here. `tools/eval/roformer_hybrid.py`.

BS-RoFormer viperx (`ep_317_sdr_12.9755`, 159.8M parameters against htdemucs's
42M) separates **one** stem: vocals, with the instrumental being the remainder.
That does not fit this server on its own: the player rebuilds `drums` from
`mix − harmonics − vocals`, and a vocals/instrumental split leaves nothing to
rebuild from. So it has to be paired, and *how* it is paired turns out to matter
as much as the model:

| config | drums | harmonics | vocals | mean |
|---|---|---|---|---|
| `htdemucs`: Fast today | 8.04 | 10.34 | 8.72 | 9.03 |
| `htdemucs_ft`: Balanced today | 8.27 | 10.88 | 9.26 | 9.47 |
| RoFormer + htdemucs, harmonics raw | 5.70 | 10.34 | 11.12 | 9.05 |
| RoFormer + htdemucs, harmonics derived | 8.10 | 11.32 | 11.12 | 10.18 |
| **RoFormer + htdemucs_ft drums** | **9.10** | **11.96** | **11.12** | **10.73** |

**The vocals gain is real**: 11.12 dB against 8.72 for Fast and 9.26 for
Balanced, an order of magnitude above this benchmark's ~0.2 dB noise floor.

**The obvious arrangement throws it away.** Shipping `harmonics` as htdemucs's
raw `bass + other` leaves the player deriving
`drums = mix − harmonics − vocals`, which evaluates to htdemucs's drums *plus
the two models' disagreement about the vocals*, that is, plus exactly the error
RoFormer was brought in to remove. Drums falls from 8.04 to 5.70 and the mean
ends up no better than plain htdemucs. The fix is to ship `harmonics` as the
remainder instead, `mix − vocals − drums`, so the player's subtraction returns
the drums estimate unchanged.

**The drums half should come from the bag's specialist.** `htdemucs_ft` is four
models and only one of them produces drums, so taking drums from it costs one
forward, not four, the same as Fast, and is worth 1.0 dB on drums and 0.6 dB
on harmonics over plain htdemucs. Two forwards in total, against Balanced's
four.

So the tier is: **vocals from BS-RoFormer, drums from `htdemucs_ft`'s drums
model, harmonics as the remainder.** +1.70 dB on the mean over Fast, +1.26 dB
over Balanced, and every one of the three parts is better than either.

Two things this does not settle. Speed is not measured here, these are padded
6.8 s clips and RoFormer is 3.8x the parameters, so what it costs on a real
track is unknown until it is ported. And the residual now lands on `harmonics`
rather than on the rebuilt part, which is the placement [the section
below](#which-part-to-rebuild) argues against on listening grounds that SDR
cannot see; harmonics scores *better* here, but that is not the same claim.

### The same arrangement, applied to Balanced

The hybrid's finding is not about RoFormer. What it says is that shipping
`harmonics` as the remainder keeps the model's disagreement with itself off the
fader the player rebuilds, and `htdemucs_ft` disagrees with itself more than
anything else here, by about 14 dB, because its four sources come from four
different models.

So the same measurement, with both halves from the bag rather than one from
each artefact:

| config | drums | harmonics | vocals | mean | models run |
|---|---|---|---|---|---|
| `htdemucs_ft`, harmonics = `bass + other` | 8.27 | 10.88 | 9.26 | 9.47 | 4 |
| the same, without the drums model | 8.27 | 10.88 | 9.26 | 9.47 | 3 |
| **drums + vocals, harmonics the remainder** | **9.10** | **11.12** | 9.26 | **9.83** | **2** |

The middle row is the same numbers as the first, which is the point of it:
Balanced never needed its drums model for anything it ships. It computed it to
report `model_residual_db` and threw the rest away.

The last row is half the work and better on every part that differs. On a 120 s
track it took Balanced from 25.8 s to 14.5 s, 4.6x realtime to **8.3x**, and
the vocals are untouched because they come from the same specialist either way.

Its `model_residual_db` is no longer measurable, for the same reason the
hybrid's is not: the three parts are now constructed to sum, so there is no
redundancy left to measure. That number reads about -160 dB on both, and means
"the arithmetic is right" rather than "the model is".

**Fast was left alone, having been measured too.**

| config | drums | harmonics | vocals | mean |
|---|---|---|---|---|
| `htdemucs`, harmonics = `bass + other` | 8.04 | 10.34 | 8.72 | 9.03 |
| the same, harmonics as the remainder | 8.10 | 10.33 | 8.72 | 9.05 |

Two hundredths of a dB, which is nothing, and the mechanism says why rather
than the other way round. What the rearrangement moves is the model's
disagreement with itself, and `htdemucs` is one model whose four sources sum to
within -33.9 dB of the mix where the bag misses by -13.9. Twenty dB less to
relocate, and one forward pass either way, so there is no speed to win and no
quality to win. The [placement argument](#which-part-to-rebuild) for keeping it
on the drums stands unopposed, and it is a listening argument that this
benchmark could not have settled anyway.

### What the quality tier costs

Speed could not be known before the port and is the part that did not go well.

| | 120 s track | realtime |
|---|---|---|
| Fast, `htdemucs` | 6.5 s | 18.4x |
| Balanced, `htdemucs_ft` | 14.5 s | 8.3x |
| **Hybrid** | **69.1 s** | **1.7x** |

Nearly five times the cost of Balanced for 1.26 dB, and about three and three
quarter minutes on a 6:38 track. Defensible for a tier that was never going to
be live, but it is the trade rather than a footnote to it.

That multiple was two and a half when this was first measured, against a
Balanced that ran four models for 25.8 s. Halving Balanced's work did not make
the quality tier worse; it made the thing it is compared against better, which
is the more honest way to read the row.

It is also genuine. The obvious suspicion was the band split and the mask
estimators: sixty-two small matmuls each, one per band, which is the shape of
launch overhead rather than work. Measured on one chunk, they are 85 ms of
4.3 s; the transformer stack is 4.7 s of it. Twelve blocks of two transformers
over 801 time steps and 62 bands at width 512 is about 8.5 TFLOP for eight
seconds of audio, and htdemucs is fifteen times less. Unlike the two wins
earlier in this document, there is nothing structural sitting in it.

Half precision is worth 1.33x here as it is everywhere else, and had to be
wired through before any of these numbers meant anything: the config declared a
dtype and the forward never cast to it, so it cost nothing and did nothing.

It costs the vocals **−54.8 dB** against the same arrangement in float32, which
is where htdemucs lands too (−53.9 dB). That was worth measuring rather than
assuming: every null test behind the port runs in float32, and this model is 24
attention layers deep against htdemucs's 5, which is the shape of thing where
half-precision error accumulates. It does not. `what_half_precision_costs_the_roformer`.

### What the tier looks like from outside the benchmark

The gain is real and it does not feel like 1.86 dB, which is worth explaining
rather than leaving someone to wonder whether the model is wired up at all.
Four 45-second excerpts, Quality against Balanced, vocals stem:

| track | Quality rms | Balanced rms | null |
|---|---|---|---|
| sung vocal throughout | 0.1346 | 0.1299 | −8.3 dB |
| vocal sample over techno | 0.0419 | 0.0349 | −8.1 dB |
| sparse vocal chops | 0.0116 | 0.0044 | −0.5 dB |
| **no vocals at all** | **0.0000** (peak 1e-4) | 0.0113 (**peak 0.499**) | +77.4 dB |

**Where both models find a vocal, they mostly agree**: about −8 dB, so a
seventh of the energy differs. The vocal is in the same place with the same
words; RoFormer wins on what it leaves behind, not on what it pulls out. That
is audible as less music bleeding through, not as a louder or clearer vocal,
and it is what +1.86 dB of SDR looks like from the fader.

**The visible win is the last row, and MUSDB could not have found it**: every
track in MUSDB has vocals. On an instrumental, `htdemucs_ft`'s vocals
specialist puts a half-scale signal into a stem that should be empty.
BS-RoFormer returns 1e-4. A player pulling the vocals fader on an instrumental
gets music from one and silence from the other, which is a larger difference in
practice than anything the benchmark scored.

Four excerpts from one library is not a benchmark and does not re-derive the
SDR above. It establishes that the tier runs the model it claims to: two
architectures cannot null at −8 dB by accident, and what the difference sounds
like on material the benchmark does not contain.
`the_quality_tier_does_not_quietly_ship_balanced_vocals`.

### Chaining the halves, and why only this tier gets it

The two halves ran side by side: both saw the track, and the only thing joining
them was the subtraction at the end. Handing the drums half `mix − vocals`
instead is free, same two forward passes, one subtraction between them, and a
model that no longer has to reject the vocals should leave fewer of them in its
drums.

It cannot touch the vocals. Those are the same forward pass over the same audio
either way, bit for bit, and the test asserts exactly that rather than nulling
it. All chaining moves is where the line between drums and harmonics falls,
and since `harmonics = mix − vocals − drums`, whatever bleeds into drums appears
*inverted* in harmonics. This is a harmonics fix as much as a drums one.

Two figures, and they are not the same one twice. **α** is `<d,v>/<v,v>`, the
least-squares amount of the vocals stem inside the drums, and is directional.
The **dB** is `<d,v>² / (|d|²|v|²)`: squared cosine similarity, which is
*symmetric*: it says the two stems share less content without saying whose the
content was. Both move together here, which is the useful case.

| track | side by side | chained |
|---|---|---|
| sung vocal throughout | −29.6 dB (α 0.063) | **−34.2 dB** (α 0.038) |
| sparse vocal chops | −44.8 dB (α 0.113) | **−52.8 dB** (α 0.045) |
| vocal sample over techno | −30.0 dB (α 0.129) | **−34.9 dB** (α 0.074) |
| no vocals at all | −35.5 dB | −35.5 dB (a no-op, correctly) |

**A cascade is worth what the model at the front of it is worth**, which is the
part worth keeping. The same change applied to `Balanced`, whose vocals half is
a demucs specialist rather than the RoFormer, measured *worse* every time:

| track | side by side | chained |
|---|---|---|
| sung vocal throughout | −28.2 dB (α 0.077) | −27.3 dB (α 0.086) |
| sparse vocal chops | −45.4 dB (α 0.276) | −43.9 dB (α 0.330) |
| vocal sample over techno | −31.0 dB (α 0.138) | −29.1 dB (α 0.172) |

What Balanced hands on is not a clean instrumental, it is an instrumental with
one demucs model's mistakes already subtracted out of it, and the next model
inherits them. So `Quality` chains and `Balanced` does not, and the flag exists
rather than the behaviour being unconditional.

There is a third arrangement this does not take: let demucs produce `harmonics`
directly, as `bass + other` over the instrumental, and let `drums` be the
remainder. It makes harmonics a model output rather than a bucket, at the cost
of a third forward pass, but the leftover is by definition everything that is
neither vocals nor drums, which is what harmonics *is*, so the bucket is the
right place for it. `chaining_the_halves_against_running_them_side_by_side`.

### What the three sound like, and the number that agrees

Listening on electronic material, with the arrangements above in place:

| preset | heard |
|---|---|
| Fast | compression and sidechain artefacts, audible tick residues |
| Balanced | the same, with less of the compression |
| Quality | the same again, with the vocal isolation clean |

"Tick residues" is a specific claim: percussive transients arriving on the
vocals fader, and it is measurable. Shared content between the vocals stem and
the drums the player rebuilds, as squared cosine similarity, on a 45 s vocal
excerpt:

| preset | vocals rms | shared with drums |
|---|---|---|
| Fast | 0.1381 | −25.3 dB |
| Balanced | 0.1299 | −28.2 dB |
| Quality | 0.1346 | **−34.2 dB** |

The order is the listening order and the steps are about 3 dB and 6 dB. Quality
leaves 9 dB less content shared between the two stems than Fast, which is the
tick residue not being there.

This is the one place in this document where an ear and a number were compared
without either being fitted to the other: the ranking was reported before the
measurement was written. It does not make the metric a proxy for quality in
general; a stem can be uncorrelated with the drums and still be wrong. It does
mean that on this material the thing being heard and the thing being measured
are the same thing. `what_each_preset_leaves_in_the_vocals`.

## BS PolarFormer: the same quality for a quarter of the arithmetic

Published leaderboards put BS PolarFormer at 11.00 multisong SDR on vocals
against the viperx checkpoint's 10.87: a 0.13 dB edge, which is *below* this
benchmark's ~0.2 dB noise floor and so not something it can resolve. Measured
anyway, on the same 25 clips and the same global SDR as everything above:

| | vocals SDR |
|---|---|
| BS-RoFormer viperx `ep_317`: what Quality runs | 11.12 |
| BS PolarFormer | **11.04** |

Ours comes out 0.08 dB ahead where the leaderboard has it 0.13 dB behind. Both
gaps are inside the noise, on different test sets. **They are the same model
for quality purposes**, and no amount of re-running settles which is nominally
better.

That is not why it is interesting. The two are the same architecture: the same
62-band table summing to 1025, depth 12, 8 heads of 64, `ff_mult` 4, the same
mask estimator shape: differing in three config values:

| | viperx | PolarFormer |
|---|---|---|
| `dim` | 512 | **256** |
| hop | 441 | 512 |
| chunk | 352,800 (8.0 s) | 588,800 (13.35 s) |
| positional | RoPE | **PoPE** |
| parameters | 159.8M | 51.1M |

### How much faster: 1.82x, measured

Both configurations built with random weights: speed does not depend on their
values, which is what makes this measurable without a second 600 MB download,
and raced on one device at one precision, over the same duration of audio
rather than the same chunk count:

| | per chunk | per second of audio | |
|---|---|---|---|
| viperx `ep_317` | 13.59 s | 1.699 s | 0.6x realtime |
| **BS PolarFormer** | 12.49 s | **0.935 s** | 1.1x realtime |

**1.82x faster.** torch/MPS at float32, so the times are not stemd's; the ratio
is the part that transfers. (float16 was the intent and MPS asserts inside a
matmul on this path, so both were timed at float32 instead: equally.)

**The first version of this document said 4.64x, from arithmetic, and it was
wrong.** Counting the multiply-accumulates again with the term that was left
out:

| per second of audio | proj + ff | attention | total |
|---|---|---|---|
| viperx | 0.468T | 0.066T | 0.534T |
| BS PolarFormer | 0.101T | 0.079T | 0.180T |

Projections and feed-forward do go as `dim²`, and there PolarFormer really is
4.64x cheaper. **Attention does not shrink with `dim` at all**: it goes as
`seq² · heads · dim_head`, and `heads · dim_head` is 512 in *both* models:
PolarFormer buys its narrower residual stream while keeping the same attention
width. Worse, its chunk is 1150 frames against 800, so its attention costs
1.2x *more* per second of audio. Corrected, the arithmetic says 2.96x; the
stopwatch says 1.82x, and the rest is kernel launches that cost the same
however narrow the matmul behind them.

**What that means for the tier.** Quality runs at 1.7x realtime, so 1.82x puts
it near 3.1x: a 6:38 track going from 235 s to about 130 s. Halved, and still
more than twice the 50 s budget the `⚠` mark exists for. That is a real
improvement for identical quality, and it is not the transformative one the
first number suggested.

### Ported, and what it actually bought: 1.2x

PoPE turned out to be the only code needed, and it nulled first try. The port
adds one struct and one branch in `Attention`; `dim`, `hop`, `chunk` and the
band table were already config, so `Config::polarformer()` is four values and
`Config::of()` picks between them by asking the tensors. Every stage of the
existing suite runs against either artefact:

```text
spectrogram  -124.9 / -124.8 dB      attention    -119.6 dB
band split   -128.0 dB               feed-forward -125.2 dB
twelve blocks -109.1 dB              mask          -95.9 dB
whole model   -88.7 dB
```

Then the whole tier on real tracks, both artefacts through the same
arrangement, 45 s excerpts:

| | seconds | vocals rms | shared with drums |
|---|---|---|---|
| viperx, techno with a vocal sample | 28.5 | 0.0419 | −30.0 dB |
| **PolarFormer**, same | **24.4** | 0.0435 | −29.3 dB |
| viperx, sung vocal | 28.5 | 0.1346 | −29.6 dB |
| **PolarFormer**, same | **23.0** | 0.1346 | −30.2 dB |

**That table says indistinguishable and it is measuring the wrong stem.** The
tick-residue figure moves 0.7 dB one way on one track and 0.6 dB the other way
on the next, and the two models' vocals share −0.2 to −0.4 dB of content. On
that evidence this section originally concluded there was nothing to choose
between them.

Listening says otherwise, specifically: viperx *leaks some voice into
harmonics*. That is a different stem from the one above. Vocals-against-drums
asks whether percussion got into the vocals; it cannot see vocal getting into
the harmonics, and a model can be clean on one and not the other.

It is measurable without ground truth, because there are two estimates of the
vocal and each is a probe for what the other left behind. If a model
under-extracts, the vocal it missed stays in its harmonics, and the other
model's vocals stem is the closest available template for what that missing
vocal looks like:

| harmonics holds, of the other model's vocals | |
|---|---|
| **techno, chopped vocal**: viperx | −39.0 dB |
| **techno, chopped vocal**: PolarFormer | **−52.5 dB** |
| sung vocal: viperx | −21.2 dB |
| sung vocal: PolarFormer | −21.5 dB |

**13.5 dB less vocal left in the harmonics on electronic material**, and no
difference at all on a conventional sung vocal. That is not a wash, and MUSDB
could not have found it: MUSDB is pop and rock, and the case where these two
models diverge is a processed, chopped vocal over techno, which is most of
what this is actually used on.

The lesson is about the metric, not the model. A number that was chosen to
settle one question (the cascade, which moves percussion) was reached for again
to settle a different one, and it answered the first question both times.

**Speed is 1.17–1.24x, not the 1.82x torch measured.** Some of that is the
demucs half, which is identical in both and about three seconds of every run.
Most of it is self-inflicted: polar doubles the depth of q and k while leaving
`v` alone, so this path cannot use `fast::scaled_dot_product_attention` and
writes out an explicit frames-by-frames score matrix that the fused kernel would
have kept in registers: at 1150 frames rather than 800. The reference gets a
fused PoPE kernel on CUDA and falls back to the same manual path elsewhere,
which is presumably why torch flattered it less than the arithmetic and more
than this does.

So the honest ledger, five numbers for one question: **4.64x** was arithmetic
with a term missing, **2.96x** was the arithmetic corrected, **1.82x** was torch,
**1.13–1.28x** is a 45 s excerpt here, and **1.05x** is a whole 6:38 track
(268 s against 256 s, of which about 25 s is the demucs half in both). Each
measurement was more careful than the last and each came in lower. Run-to-run
spread on the excerpts is ±5%, so the last two are the same number.

**Speed is a wash. Quality is not, on this material.** That is the reverse of
what the first pass concluded, and the reversal came from an ear, not from a
better benchmark.

The remaining speed gap has a name and a fix: a fused attention that tolerates
`v` being narrower than `q`, but it is no longer the reason to care about this
model.

**`Quality` runs this one now** and `bs_roformer_viperx` is retired. It was not
replaced for being wrong, it is a faithful port, nulled stage by stage, and it
carried this tier from the day it existed. It was replaced because something
leaves less voice in the harmonics for the same money, and because that turned
out to be the thing worth measuring.

`Config::of` still tells the two artefacts apart, so the null suite and the A/B
run against either; the viperx tests need the retired file, which the app no
longer fetches.

`tools/eval/polarformer.py` for quality, `tools/eval/roformer_speed.py` for
speed. Both need ZFTurbo's Music-Source-Separation-Training on the path for the
model class, and `PoPE-pytorch` installed.

## Tuning levers

| lever | quality | cost | verdict |
|---|---|---|---|
| overlap 0.50 | ~+0.03 dB | 30 s | below noise |
| shifts=1 / shifts=2 | −0.07 / −0.02 dB | 2x | negative |
| ensemble with htdemucs | +0.11 dB | 51 s | over budget |
| which part is rebuilt | ±0.04 dB |: | noise |
| soft-mask refinement | **−1.55 dB** |: | wrecks quality |
| fp16 trunk | −62.5 dB error vs the mix (0.01 dB of the budget) | 10% faster, artefact halves | not worth a second artefact |

Soft-mask refinement (`S_i = mix · |S_i|^a / Σ|S_j|^a`) is the standard
spectrogram-domain trick and costs 1.5 dB here. demucs works in the waveform
domain and recovers phase; masking substitutes the mix's phase. demucs dropped
its own Wiener option for the same reason.

**The spare budget cannot be converted into quality.** Nothing available beats
the current configuration by more than 0.11 dB, and the only positive entry costs
more than the whole budget. Two independently trained SOTA models disagree with
each other by **−22 dB of the mix** on drums and harmonics; that misassignment is
the dominant error and no knob here touches it. Everything tunable sits 14 dB or
more below it: sum-residual at −35 dB, overlap convergence at −36 dB.

Shipping all three model parts scores 9.48 and rebuilding one scores 9.44–9.48,
so **exact reconstruction is very nearly free**.

## Segment length

Measured when `--segment` was a trace-time parameter baked into a TorchScript
artefact. It no longer is: the MLX port runs the model's native segment, because
htdemucs's position embedding was learned at that extent and a shorter input is a
different model rather than a faster one. Kept because the conclusion still
holds: a larger segment costs more than proportionally.

| model | segment | wall (6:38) | RTF |
|---|---|---|---|
| hdemucs_mmi | 44 s (native) | **19.2 s** | 20.7x |
| htdemucs | 7.8 s (native) | 43.9 s | 9.1x |
| htdemucs | 10 s | 46.2 s | 8.6x |

A 398 s track needs 13 passes at 44 s and 68 at 7.8 s, so htdemucs does 5.2x the
passes for 2.29x the time and is cheaper per pass. Overriding its segment to 10 s
cuts the count by 28% and runs **5% slower**: a larger segment costs more than
proportionally, so the trade is a net loss.

Anything timed above load ~8 on this hardware is noise.

## Runtime (hdemucs_mmi, on TorchScript)

| stage | RTF | 5-min track |
|---|---|---|
| inference alone | 44.1x | 6.8 s |
| full server round trip | 21.6x | ~14 s |

The gap is single-threaded DSP around the model: STFT, two inverse STFTs,
residual, encode, file writes. Threading it would contend with the pool already
using every core; throughput under load is the queue's job.

End to end from the player: mDNS discovery, 5.3 MB upload, separation, 15.9 MB
of stems back: is **3 s for 30 s of audio** over its gigabit link (MTU 1200).

## Which part to rebuild

The two transferred stems are raw model output whatever the choice. Only the
rebuilt part differs, it is the model's version plus the whole residual, so the
question is which stem absorbs the model's error. SDR cannot answer it: the three
choices land within ±0.04 dB.

Measured over 90 s of real material, htdemucs. Residual −29.8 dB of the mix, crest
factor **14.3** against the mix's 2.5: impulsive, so it sounds percussive.

| rebuilt | its level | residual below it | drums dropped | solo'd |
|---|---|---|---|---|
| **drums** | −3.7 dB | 26.0 dB | **clean** | percussive error in a percussive stem |
| harmonics | −2.6 dB | 27.2 dB | ghost hits, −27.3 dB RMS | ghost hits over pads and bass |
| vocals | −31.3 dB | **−2.6 dB** | ghost hits, −27.3 dB RMS | mostly model error |

Two independent reasons for `drums`:

- **Level.** On instrumental-leaning material the vocals stem can sit *below* the
  residual, which would make that gain mostly model error. Drums and harmonics
  are reliably loud; vocals is the one that can approach silence.
- **Placement.** Zeroing a gain removes the residual with it. Only rebuilding
  drums makes the drums control remove the model's percussive error: otherwise
  it survives on a stem that is still playing, at −27.3 dB RMS whose *peaks* reach
  about −4 dB of the remaining material at that crest factor.

Harmonics is fine on level and fails on placement. It would be defensible only
where the drums gain is never pulled.

## On-device separation

Measured on an RK3399 client (4x Cortex-A53 @1.416 GHz + 2x Cortex-A72
@1.608 GHz), for reference on why separation is offloaded:

| | fp32 NEON FMA |
|---|---|
| M1 Firestorm, 1 core | 50.45 GFLOP/s |
| A72 (cpu5) | 4.06–4.33 GFLOP/s |
| A53 (cpu0-3), each | 0.78–0.92 GFLOP/s |
| **aggregate, 5 usable cores under load** | **7.52 GFLOP/s** |

One A72 is fenced off by `isolcpus=4 nohz_full=4 rcu_nocbs=4` for an audio RT
thread. Hybrid Demucs v3 costs **2.31 s of compute per 1 s of audio** there: 63%
Conv, 12% ConvTranspose, 12% LSTM. Deleting the entire LSTM bottleneck buys 1.13x,
the whole time-domain branch 1.29x, both 1.52x. The gap to real time is ~5x and
the removable structure is worth 1.5x.

## Benchmark limits

MUSDB clips are 6.8 s, shorter than a 44 s segment, so the benchmark cannot see
segment boundaries and cannot measure `--overlap`; sweep that against a real
track through the server. 25 clips cannot resolve differences under ~0.2 dB,
which is where most levers landed.
