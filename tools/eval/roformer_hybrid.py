"""Does BS-RoFormer's vocals actually earn the port? Measured: yes.

Run before porting BS-RoFormer to stemd-mlx, to find out whether the model was
worth the work before doing the work. It was, and the arrangement matters as
much as the model -- see docs/evaluation.md.

Needs three things this repo does not carry:

  * the checkpoint, model_bs_roformer_ep_317_sdr_12.9755.ckpt (639 MB), from
    Sucial/MSST-WebUI on Hugging Face under All_Models/vocal_models/
  * its config of the same name, from ZFTurbo/Music-Source-Separation-Training
    under configs/viperx/
  * that repo's models/bs_roformer/ on PYTHONPATH. The published `bs-roformer`
    package has moved on -- it grew hyper-connections -- and no longer loads
    these weights; the vendored copy is the code they were trained with, and it
    loads them strictly, no missing and no unexpected tensors.

    uv run tools/eval/roformer_hybrid.py --ckpt ... --config ... --clips 25

Scores the three parts the player drives, on the same MUSDB clips and the same
global SDR (MDX/multisong convention) as tools/eval/benchmark.py, for:

  htdemucs        what Fast does today: all four sources from one model
  htdemucs_ft     what Balanced does today
  hybrid-clean    vocals from BS-RoFormer, harmonics = mix - vocals - drums
                  with drums from htdemucs, so the part the player rebuilds
                  comes out exactly htdemucs's raw drums
  hybrid-derived  vocals from BS-RoFormer, harmonics = htdemucs's bass + other,
                  so the player's drums is htdemucs's drums PLUS the two
                  models' disagreement about the vocals

Both are worth measuring because they differ in where the model's unexplained
residual lands, and the project has an opinion about that: docs/evaluation.md
argues for drums, because the residual is impulsive and hides in a percussive
stem while it reads as ghost hits over pads and bass. `hybrid-clean` quietly
moves it to harmonics, which is the arrangement that document rejected. So the
question is whether the vocals gained are worth the placement lost.
"""

import argparse
import sys
import time

import numpy as np
import torch
import yaml
import musdb
from demucs.apply import apply_model
from demucs.pretrained import get_model

from models.bs_roformer.bs_roformer import BSRoformer

PARTS = {"drums": ["drums"], "harmonics": ["bass", "other"], "vocals": ["vocals"]}
SOURCES = ["drums", "bass", "other", "vocals"]


def global_sdr(ref, est):
    num = float((ref.astype(np.float64) ** 2).sum())
    den = float(((ref - est).astype(np.float64) ** 2).sum())
    if den == 0 or num == 0:
        return np.nan
    return 10 * np.log10(num / den)


def load_roformer(ckpt, config, device):
    import inspect

    cfg = yaml.load(open(config), Loader=yaml.UnsafeLoader)
    accepted = set(inspect.signature(BSRoformer.__init__).parameters)
    model = BSRoformer(**{k: v for k, v in cfg["model"].items() if k in accepted})
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval().to(device)
    return model, int(cfg["audio"]["chunk_size"])


def roformer_vocals(model, chunk, mix, device):
    """`mix` is [C, T]; returns the vocals estimate at the same shape.

    Clips are shorter than one chunk, so this pads to the chunk the model was
    trained on and trims back -- the same thing apply_model does for demucs.
    """
    length = mix.shape[1]
    padded = np.zeros((mix.shape[0], max(chunk, length)), dtype=np.float32)
    padded[:, :length] = mix
    with torch.no_grad():
        out = model(torch.from_numpy(padded)[None].to(device))
    out = out.squeeze(0).cpu().numpy()
    if out.ndim == 3:  # [stems, C, T] with num_stems == 1
        out = out[0]
    return out[:, :length]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--clips", type=int, default=25)
    ap.add_argument("--overlap", type=float, default=0.25)
    args = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    db = musdb.DB(download=True)
    tracks = db[: args.clips]

    demucs = {name: get_model(name).eval() for name in ("htdemucs", "htdemucs_ft")}
    roformer, chunk = load_roformer(args.ckpt, args.config, device)

    configs = ["htdemucs", "htdemucs_ft", "hybrid-clean", "hybrid-derived", "hybrid-ft-drums"]
    scores = {c: {p: [] for p in PARTS} for c in configs}
    wall = {c: 0.0 for c in configs}
    audio_secs = 0.0

    print(
        f"{len(tracks)} MUSDB clips | {device} | overlap {args.overlap} | "
        f"roformer chunk {chunk / 44100:.1f}s\n"
    )

    for n, track in enumerate(tracks):
        mix = track.audio.T.astype(np.float32)
        audio_secs += mix.shape[1] / track.rate
        truth = {
            p: sum(track.targets[s].audio.T.astype(np.float32) for s in members)
            for p, members in PARTS.items()
        }

        raw = {}
        took_demucs = 0.0
        for name, m in demucs.items():
            t0 = time.time()
            with torch.no_grad():
                out = apply_model(
                    m, torch.from_numpy(mix)[None], device=device,
                    shifts=0, split=True, overlap=args.overlap, progress=False,
                )[0]
            elapsed = time.time() - t0
            wall[name] += elapsed
            if name == "htdemucs":
                took_demucs = elapsed
            raw[name] = {s: out[m.sources.index(s)].cpu().numpy() for s in SOURCES}

        t0 = time.time()
        vocals_r = roformer_vocals(roformer, chunk, mix, device)
        roformer_secs = time.time() - t0
        demucs_secs = took_demucs

        for name in ("htdemucs", "htdemucs_ft"):
            est = {p: sum(raw[name][s] for s in members) for p, members in PARTS.items()}
            est["drums"] = mix - est["harmonics"] - est["vocals"]
            for p in PARTS:
                v = global_sdr(truth[p], est[p])
                if np.isfinite(v):
                    scores[name][p].append(v)

        drums_h = raw["htdemucs"]["drums"]
        harmonics_h = raw["htdemucs"]["bass"] + raw["htdemucs"]["other"]

        # Ship harmonics as the remainder: the player's derived drums comes back
        # as htdemucs's raw drums, and harmonics carries the residual.
        clean = {
            "vocals": vocals_r,
            "harmonics": mix - vocals_r - drums_h,
            "drums": drums_h,
        }
        # Ship harmonics raw: the residual stays on the derived drums, where the
        # project wants it, along with the two models' vocal disagreement.
        derived_drums = mix - harmonics_h - vocals_r
        naive = {
            "vocals": vocals_r,
            "harmonics": harmonics_h,
            "drums": derived_drums,
        }
        # The same as hybrid-clean but taking drums from htdemucs_ft's drums
        # specialist. In production that is one forward, not four: only the
        # model whose weight column is non-zero has to run.
        drums_ft = raw["htdemucs_ft"]["drums"]
        specialist = {
            "vocals": vocals_r,
            "harmonics": mix - vocals_r - drums_ft,
            "drums": drums_ft,
        }
        for cfg, est in (
            ("hybrid-clean", clean),
            ("hybrid-derived", naive),
            ("hybrid-ft-drums", specialist),
        ):
            for p in PARTS:
                v = global_sdr(truth[p], est[p])
                if np.isfinite(v):
                    scores[cfg][p].append(v)
            wall[cfg] += roformer_secs + demucs_secs

        print(f"  clip {n + 1}/{len(tracks)}", end="\r", flush=True)

    print(" " * 30, end="\r")
    header = f"  {'config':<14}" + "".join(f"{p:>11}" for p in PARTS) + f"{'mean':>9}{'speed':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cfg in configs:
        per = [float(np.mean(scores[cfg][p])) for p in PARTS]
        rt = audio_secs / wall[cfg] if wall[cfg] else float("inf")
        print(
            f"  {cfg:<14}" + "".join(f"{v:>11.2f}" for v in per)
            + f"{np.mean(per):>9.2f}{rt:>11.2f}x"
        )
    print(
        "\n  speed is on padded clips and pessimistic for every row; the point "
        "here is\n  the dB, not the x. Differences under ~0.2 dB are inside the "
        "noise of this\n  many clips."
    )


if __name__ == "__main__":
    sys.exit(main())
