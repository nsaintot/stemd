"""Score every preset per part through a running stemd-server and stemd-cli.

Full tracks, global SDR, on the first 12 MUSDB18-HQ test tracks.
`stemd_roformer` was not trained on them. Starts one headless server per preset.

    uv run --extra eval --with soundfile tools/eval/presets.py \\
        --bin <dir with stemd-server and stemd-cli> --musdb <musdb18-hq/test>

Each track is separated twice:

| request | scored parts |
| --- | --- |
| default (`harmonics,vocals`) | `drums` (rebuilt from the mix), `harmonics`, `vocals` |
| `bass,other,vocals` | `bass`, `other` |
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request

import numpy as np
import soundfile as sf

PRESETS = [("Fast", "htdemucs"), ("Balanced", "htdemucs_ft"), ("Quality", "stemd_roformer")]
PARTS = ("drums", "bass", "other", "harmonics", "vocals")
SOURCES = ("bass", "drums", "other", "vocals")


def read(path):
    return sf.read(path, dtype="float64", always_2d=True)[0].T


def gsdr(ref, est):
    return 10 * np.log10((ref**2).sum() / ((ref - est) ** 2).sum())


def tracks(root, count):
    names = sorted(os.listdir(root))
    found = [
        os.path.join(root, n)
        for n in names
        if all(os.path.exists(os.path.join(root, n, f"{s}.wav")) for s in SOURCES + ("mixture",))
    ]
    return found[:count]


def wait_ready(proc, host, timeout=900):
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"stemd-server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://{host}/v1/health", timeout=5) as r:
                health = json.load(r)
            if health.get("model_id"):
                return health
        except OSError:
            pass
        time.sleep(3)
    raise RuntimeError("stemd-server did not answer /v1/health")


def separate(cli, host, mix, out, parts=None):
    shutil.rmtree(out, ignore_errors=True)
    cmd = [cli, mix, "--host", host, "--f32", "--wav-f32", "--out", out]
    if parts:
        cmd += ["--parts", parts]
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if run.returncode != 0:
        raise RuntimeError(f"stemd-cli failed: {run.stderr.strip()[:400]}")
    m = re.search(r"\(([\d.]+)x realtime\)", run.stdout)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="directory with stemd-server and stemd-cli")
    ap.add_argument("--musdb", required=True, help="MUSDB18-HQ test directory")
    ap.add_argument("--out", default="presets-out")
    ap.add_argument("--tracks", type=int, default=12)
    ap.add_argument("--host", default="127.0.0.1:8421")
    args = ap.parse_args()

    server = os.path.join(args.bin, "stemd-server")
    cli = os.path.join(args.bin, "stemd-cli")
    env = dict(os.environ)
    bundled_cuda = os.path.join(args.bin, "..", "cuda")
    if "CUDA_HOME" not in env and os.path.isdir(bundled_cuda):
        # An extracted AppImage or .deb tree: its launcher sets this.
        env["CUDA_HOME"] = os.path.abspath(bundled_cuda)
    os.makedirs(args.out, exist_ok=True)
    results = {}
    for label, model in PRESETS:
        cache = os.path.join(args.out, "cache")
        shutil.rmtree(cache, ignore_errors=True)
        with open(os.path.join(args.out, f"server-{label}.log"), "w") as log:
            proc = subprocess.Popen(
                [server, "--headless", "--no-mdns", "--bind", args.host,
                 "--demucs-model", model, "--cache-dir", cache, "--cache-max-gb", "40",
                 "--max-track-minutes", "15",
                 "--settings", os.path.join(args.out, "settings.json")],
                env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                health = wait_ready(proc, args.host)
                print(f"== {label}: {health['model_id']}", flush=True)
                rows, speed = [], []
                for track in tracks(args.musdb, args.tracks):
                    mix = os.path.join(track, "mixture.wav")
                    a, b = os.path.join(args.out, "a"), os.path.join(args.out, "b")
                    rt = separate(cli, args.host, mix, a)
                    separate(cli, args.host, mix, b, "bass,other,vocals")
                    truth = {s: read(os.path.join(track, f"{s}.wav")) for s in SOURCES}
                    truth["harmonics"] = truth["bass"] + truth["other"]
                    est = {s: read(os.path.join(a, f"{s}.wav"))
                           for s in ("drums", "harmonics", "vocals")}
                    est.update({s: read(os.path.join(b, f"{s}.wav")) for s in ("bass", "other")})
                    k = min(v.shape[1] for v in list(truth.values()) + list(est.values()))
                    row = {s: gsdr(truth[s][:, :k], est[s][:, :k]) for s in PARTS}
                    rows.append({"track": os.path.basename(track), **row})
                    if rt:
                        speed.append(rt)
                    print(f"  {os.path.basename(track)[:36]:<36} "
                          + " ".join(f"{s} {row[s]:6.2f}" for s in PARTS), flush=True)
                mean = {s: float(np.mean([r[s] for r in rows])) for s in PARTS}
                results[label] = {"model_id": health["model_id"], "mean": mean, "rows": rows,
                                  "realtime": float(np.mean(speed)) if speed else None}
                print(f"RESULT {label:<9} " + " ".join(f"{s} {mean[s]:.2f}" for s in PARTS),
                      flush=True)
            finally:
                proc.terminate()
                try:
                    proc.wait(30)
                except subprocess.TimeoutExpired:
                    proc.kill()
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
