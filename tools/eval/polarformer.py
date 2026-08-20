"""Score BS PolarFormer's vocals against the model stemd's Quality tier runs.

BS PolarFormer is a BS-RoFormer with polar positional embeddings instead of
rotary ones, and it is published at 11.00 multisong SDR against the viperx
checkpoint's 10.87. That difference is not the interesting part: it is 0.13 dB,
and docs/evaluation.md puts this benchmark's noise floor at ~0.2 dB, so this
script cannot resolve it and does not pretend to.

The interesting part is the cost. The two are the same architecture --
identical 62-band table, depth 12, 8 heads of 64, ff_mult 4 -- differing in
`dim`, 256 against 512, and in the chunk and hop they run at. The transformer
stack is 97% of a forward pass, so half the width is most of the way to a
quality tier that is not four minutes long.

What this measures, on the same 25 MUSDB clips and the same global SDR as
tools/eval/roformer_hybrid.py, so the numbers land in the same table:

  * vocals SDR, to check the published figure holds on this benchmark and that
    nothing is wired up wrong;
  * `harmonics` as the remainder, which is what the Quality tier ships, since a
    better vocals model moves that stem too;
  * seconds per second of audio, which is a torch/MPS number and does not
    predict MLX -- the honest cost signal is the arithmetic printed beside it.

Needs ZFTurbo's Music-Source-Separation-Training on the path for the model
class, and `PoPE-pytorch` installed.

```text
uv run --extra eval tools/eval/polarformer.py \
    --msst <path to the checkout> \
    --ckpt model_bs_polarformer_float16.ckpt \
    --config model_bs_polarformer_float16.yaml
```
"""

import argparse
import inspect
import sys
import time

import numpy as np
import torch
import yaml
import musdb

PARTS = {"drums": ["drums"], "harmonics": ["bass", "other"], "vocals": ["vocals"]}


def global_sdr(ref, est):
    """MDX/multisong convention: one ratio over the whole signal, not per frame."""
    num = float((ref.astype(np.float64) ** 2).sum())
    den = float(((ref - est).astype(np.float64) ** 2).sum())
    if den == 0 or num == 0:
        return float("nan")
    return 10.0 * np.log10(num / den)


def load(ckpt, config, device):
    from models.bs_roformer.bs_roformer import BSRoformer

    cfg = yaml.load(open(config), Loader=yaml.UnsafeLoader)
    accepted = set(inspect.signature(BSRoformer.__init__).parameters)
    dropped = sorted(set(cfg["model"]) - accepted)
    if dropped:
        print(f"config keys the class does not take, ignored: {dropped}")
    model = BSRoformer(**{k: v for k, v in cfg["model"].items() if k in accepted})
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        # Loud, because a silently half-loaded model still produces audio and
        # would score like a bad model rather than like a bug.
        print(f"WARNING missing={len(missing)} unexpected={len(unexpected)}")
        print(f"  first missing: {missing[:3]}")
        print(f"  first unexpected: {unexpected[:3]}")
    model.eval().to(device)
    return model, cfg


def vocals_of(model, chunk, mix, device):
    """`mix` is [C, T]; returns the vocals estimate at the same shape.

    The clips are shorter than one chunk, so this pads out to the chunk the
    model was trained on and trims back, exactly as roformer_hybrid.py does.
    """
    length = mix.shape[1]
    padded = np.zeros((mix.shape[0], max(chunk, length)), dtype=np.float32)
    padded[:, :length] = mix
    with torch.no_grad():
        out = model(torch.from_numpy(padded)[None].to(device))
    out = out.squeeze(0).float().cpu().numpy()
    if out.ndim == 3:  # [stems, C, T] with num_stems == 1
        out = out[0]
    return out[:, :length]


def transformer_cost(dim, depth, bands, frames, heads, dim_head, ff_mult=4):
    """Multiply-accumulates the transformer stack performs for one chunk.

    Returns `(projections_and_ff, attention)`, separately, because they scale
    with different things and the first version of this counted only the first.
    That was wrong by more than double:

    * projections and feed-forward go as `dim^2`, so halving the width is 4x;
    * attention goes as `seq^2 * heads * dim_head`, and `heads * dim_head` is
      512 in *both* of these models. **Attention does not shrink with `dim` at
      all.** It grows with the chunk, and PolarFormer's chunk is 1150 frames
      against 800, so its attention costs twice as much per chunk.

    Counting only the first term said 4.64x less. Counting both says 2.96x. A
    stopwatch says 1.82x -- see tools/eval/roformer_speed.py, and prefer it.
    """
    tokens = bands * frames
    inner = heads * dim_head
    projections = 2 * depth * tokens * (4 * dim * dim + 2 * ff_mult * dim * dim)
    # A time transformer attending over frames within each band, and a
    # frequency one attending over bands at each frame. Two matmuls each.
    over_time = bands * frames * frames * inner * 2
    over_bands = frames * bands * bands * inner * 2
    return projections, depth * (over_time + over_bands)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--msst", required=True, help="Music-Source-Separation-Training checkout")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--clips", type=int, default=25)
    args = ap.parse_args()

    sys.path.insert(0, args.msst)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model, cfg = load(args.ckpt, args.config, device)
    chunk = int(cfg["audio"]["chunk_size"])
    m = cfg["model"]

    db = musdb.DB(download=True)
    tracks = db[: args.clips]
    print(f"{len(tracks)} MUSDB clips | {device} | chunk {chunk / 44100:.2f}s\n")

    scores = {p: [] for p in PARTS}
    wall, audio_secs = 0.0, 0.0
    for n, track in enumerate(tracks):
        mix = track.audio.T.astype(np.float32)
        audio_secs += mix.shape[1] / track.rate
        truth = {
            p: sum(track.targets[s].audio.T.astype(np.float32) for s in members)
            for p, members in PARTS.items()
        }

        t0 = time.time()
        vocals = vocals_of(model, chunk, mix, device)
        wall += time.time() - t0

        # Vocals is the only thing this model determines, so it is the only
        # thing scored against the tier.
        #
        # `harmonics` is reported too but is NOT the tier's harmonics and must
        # not be read beside it: the remainder here is taken against the *true*
        # drums rather than a demucs estimate, because this script does not run
        # demucs. That makes it an upper bound -- what the tier's harmonics
        # would score if the drums half were perfect -- which is useful for
        # seeing how much of that stem this model is responsible for, and
        # useless as a comparison. It lands several dB high for that reason.
        est = {"vocals": vocals, "harmonics": mix - vocals - truth["drums"]}
        for p, e in est.items():
            v = global_sdr(truth[p], e)
            if np.isfinite(v):
                scores[p].append(v)
        print(f"  clip {n + 1}/{len(tracks)}", end="\r", flush=True)

    print(" " * 30, end="\r")
    print(f"  {'part':<12}{'SDR':>9}")
    print("  " + "-" * 21)
    print(f"  {'vocals':<12}{np.mean(scores['vocals']):>9.2f}   <- comparable")
    print(
        f"  {'harmonics':<12}{np.mean(scores['harmonics']):>9.2f}   <- upper bound, "
        f"true drums, NOT the tier's figure"
    )
    print(f"\n  {wall:.1f}s for {audio_secs:.0f}s of audio ({audio_secs / wall:.1f}x realtime, torch/MPS)")

    # Arithmetic, for the record. It is not the answer: measure with
    # tools/eval/roformer_speed.py, which races both configurations on one
    # device and lands at 1.82x where this says 2.96x.
    bands = len(m["freqs_per_bands"])
    tp, ta = transformer_cost(
        m["dim"], m["depth"], bands, chunk // m["stft_hop_length"],
        m["heads"], m["dim_head"], m["mlp_expansion_factor"],
    )
    op, oa = transformer_cost(512, 12, 62, 352_800 // 441, 8, 64, 4)
    theirs, ours = (tp + ta) / (chunk / 44100), (op + oa) / 8.0
    print("\n  transformer MACs per second of audio")
    print(f"  {'':<12}{'proj+ff':>10}{'attention':>12}{'total':>10}")
    print(f"  {'PolarFormer':<12}{tp / (chunk / 44100) / 1e12:>9.3f}T"
          f"{ta / (chunk / 44100) / 1e12:>11.3f}T{theirs / 1e12:>9.3f}T")
    print(f"  {'viperx':<12}{op / 8.0 / 1e12:>9.3f}T{oa / 8.0 / 1e12:>11.3f}T{ours / 1e12:>9.3f}T")
    print(f"\n  {ours / theirs:.2f}x less arithmetic -- but attention is {ta / oa:.1f}x *more*,")
    print("  because heads*dim_head is 512 in both and only the chunk differs.")


if __name__ == "__main__":
    main()
