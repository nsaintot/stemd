"""Score separation configurations against MUSDB ground truth.

Every model and tuning decision in this repo was settled with this, and the
numbers in the README come from it. Re-run it before changing the model, the
overlap, or which part is derived -- the interesting results were all the ones
that contradicted the reasoning.

    uv run --extra eval tools/eval/benchmark.py
    uv run --extra eval tools/eval/benchmark.py --clips 50 --overlap 0.5

It scores the three parts the player actually drives (drums / harmonics /
vocals), not the model's four raw sources, and it scores them *after* deriving
one by subtraction -- which is what the player hears. Global SDR, the
MDX/multisong convention.

Two limits worth knowing before trusting a number:

* MUSDB's sample clips are 6.8 s, shorter than the model's 44 s segment, so
  every clip is one padded segment. That makes this blind to segment-boundary
  effects and therefore blind to `--overlap`. Sweep overlap on a real track
  through the server instead.
* Differences under about 0.2 dB are inside the noise of a 25-clip sample. Most
  of the levers tried here landed there, which is itself the finding.
"""

import argparse
import time

import musdb
import numpy as np
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

SOURCES = ["drums", "bass", "other", "vocals"]
# Must match stemd_core::stems::PART_SOURCES.
PARTS = {"drums": ["drums"], "harmonics": ["bass", "other"], "vocals": ["vocals"]}


def global_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    num = float((ref.astype(np.float64) ** 2).sum())
    den = float(((ref - est).astype(np.float64) ** 2).sum())
    if den == 0 or num == 0:
        return np.nan
    return 10 * np.log10(num / den)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["hdemucs_mmi"])
    ap.add_argument("--clips", type=int, default=25)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--shifts", type=int, default=0)
    ap.add_argument("--derive", default="drums", choices=[*PARTS, "none"],
                    help="'none' ships all parts, so the sum is inexact")
    ap.add_argument("--ensemble", action="store_true",
                    help="average the models instead of scoring them separately")
    args = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    db = musdb.DB(download=True)
    tracks = db[: args.clips]

    loaded = {}
    for name in args.models:
        m = get_model(name)
        m.eval()
        loaded[name] = m

    configs = ["ensemble"] if args.ensemble else list(loaded)
    scores = {c: {p: [] for p in PARTS} for c in configs}
    wall = {c: 0.0 for c in configs}
    audio_secs = 0.0

    print(f"{len(tracks)} MUSDB clips | {device} | overlap {args.overlap} | "
          f"shifts {args.shifts} | derived {args.derive}\n")

    for track in tracks:
        mix = track.audio.T.astype(np.float32)
        audio_secs += mix.shape[1] / track.rate
        truth = {
            p: sum(track.targets[s].audio.T.astype(np.float32) for s in members)
            for p, members in PARTS.items()
        }

        raw = {}
        # Per model, not a share of the total: models differ in cost by 4x here
        # (htdemucs_ft is a bag of four), and splitting the total evenly reports
        # them as identical.
        took = {}
        for name, m in loaded.items():
            torch.manual_seed(0)  # so shifts is a comparison, not a lottery
            t0 = time.time()
            with torch.no_grad():
                out = apply_model(m, torch.from_numpy(mix)[None], device=device,
                                  shifts=args.shifts, split=True,
                                  overlap=args.overlap, progress=False)[0]
            took[name] = time.time() - t0
            raw[name] = {s: out[m.sources.index(s)].cpu().numpy() for s in SOURCES}

        if args.ensemble:
            merged = {s: np.mean([raw[n][s] for n in loaded], axis=0) for s in SOURCES}
            runs = {"ensemble": (merged, sum(took.values()))}
        else:
            runs = {n: (raw[n], took[n]) for n in loaded}

        for cfg, (srcs, secs) in runs.items():
            wall[cfg] += secs
            est = {p: sum(srcs[s] for s in members) for p, members in PARTS.items()}
            if args.derive != "none":
                est[args.derive] = mix - sum(v for p, v in est.items() if p != args.derive)
            for p in PARTS:
                value = global_sdr(truth[p], est[p])
                if np.isfinite(value):
                    scores[cfg][p].append(value)

    header = f"  {'config':<16}" + "".join(f"{p:>11}" for p in PARTS) + f"{'mean':>9}{'speed':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cfg in configs:
        per = [float(np.mean(scores[cfg][p])) for p in PARTS]
        rt = audio_secs / wall[cfg] if wall[cfg] else float("inf")
        print(f"  {cfg:<16}" + "".join(f"{v:>11.2f}" for v in per)
              + f"{np.mean(per):>9.2f}{rt:>11.2f}x")

    print("\n  speed is measured on padded 6.8 s clips and is pessimistic; time a "
          "real track\n  through the server instead. Differences under ~0.2 dB are noise.")


if __name__ == "__main__":
    main()
