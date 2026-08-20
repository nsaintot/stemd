# Building, packaging and flags

Everything a contributor needs. The [README](../README.md) is for people who
want to use the program.

## Clone

```bash
git clone --recurse-submodules https://github.com/nsaintot/stemd
cargo build --release
```

The submodule is `vendor/mlx-rs-stemd`: MLX's Rust bindings, forked to build off
Apple silicon. Nothing resolves without it, and `git submodule update --init` is
the fix for a clone that forgot. It is a fork rather than a version because
0.25.3 and 0.2.0 are the newest published `mlx-rs` and `mlx-sys`, and neither
declares a CUDA feature.

Everywhere: `cmake` on `PATH`, and a Rust new enough for edition 2024, so 1.85
or later. MLX builds from source on the first `cargo build`.

## Per platform

**macOS**, Apple silicon, Metal. The Xcode command line tools and
`brew install cmake`. About four minutes, which is what those two are for.

**Windows**, x86-64, CUDA or CPU. MSVC, the CUDA toolkit, cuDNN, and LLVM for
`libclang`. Use `scripts/bundle-windows.ps1` rather than setting the
environment by hand: it sets `MLX_BUILD_CUDA`, finds cuDNN, and then checks the
binary it produced actually carries the CUDA imports. That check exists because
a CPU-only build is silent. It links, starts, reports `device: cpu`, separates
correctly, and takes 610 seconds over a clip the card does in three.

Keep those variables identical between builds. cargo re-runs the MLX build
script when any of them moves, and that is forty minutes rather than four.

**Linux**, x86-64, CUDA. `libavahi-compat-libdnssd-dev` for the mDNS
advertisement, and the CUDA toolkit plus cuDNN 9.5 or later. Use
`scripts/bundle-linux.sh`, which checks all of it before compiling anything,
because the failures are otherwise sixty lines into a CMake trace naming
something that is not what is missing.

cuDNN must be 9.5 or later: the cudnn-frontend MLX builds against calls
`cudnnBackendPopulateCudaGraph`, which was added there. An older one compiles
and then finds no execution plan for any graph at run time. Debian's own
`nvidia-cudnn` is 9.0.0 and built for CUDA 12, so it does not qualify.

## Packaging

```bash
./scripts/bundle-app.sh          # macOS: dist/stemd.app, about 29 MB
./scripts/package-mac.sh         # macOS: a signed, notarized, stapled .dmg
./scripts/bundle-windows.ps1     # Windows: dist/stemd-windows
./scripts/package-windows.ps1    # Windows: an installer and a zip
./scripts/package-deb.sh         # Linux: a .deb
./scripts/bundle-linux.sh        # Linux: a self-contained directory, over 1 GB
```

`MLX_CUDA_ARCHITECTURES` decides which cards the result runs on, and it is the
single largest thing in the size and the build time. Unset, MLX builds for the
card in the machine, which is a package that will not start anywhere else. The
released artefacts are built with `80;86;89;90;120`, Ampere through Blackwell,
which is five compilations of every kernel: about two and a half hours on
sixteen cores, and a .deb of 199 MB rather than 55.

The Windows installer offers both modes. Just for me, into
`%LOCALAPPDATA%\Programs\stemd`, is the default; for all users goes to Program
Files. `Programs\stemd` rather than `stemd` because `%LOCALAPPDATA%\stemd` is
where the program keeps its weights and its cache, and an uninstaller pointed at
that directory would take a gigabyte of downloads with it.

CUDA is what the two modes actually differ in. `install-cuda.cmd` writes about
1.2 GB beside the executable, which a per-user install can do and Program Files
cannot, so the all-users install replaces that script with one that asks for an
administrator rather than failing on a permission error a gigabyte in.

Elevation is at launch, not mid-run: stock NSIS cannot raise its own token, and
the plugin that can is not part of NSIS. So the installer manifest asks for the
highest token available, which on an administrator account means one consent
prompt even for a per-user install. The installer needs NSIS, which
`winget install NSIS.NSIS` provides.

Two Linux artefacts on purpose. The `.deb` takes CUDA and cuDNN from the
distribution, which means one copy of each on the machine rather than two that
can disagree about which engines to load. The directory carries its own copy of
everything and exists for a machine with no package manager, such as a live USB.

Which distributions can install the `.deb` is decided by the machine that built
it, not by the packaging. Built on Debian 13 it requires glibc 2.39 and
libstdc++ 14, which Debian 13 and Ubuntu 24.04 have and Ubuntu 22.04 and Debian
12 do not. Every other dependency resolves on all four. Build on the oldest
target you mean to support.

On the macOS bundle: the icon is built from `resources/` with `sips` and
`iconutil`, both part of macOS, so a clone needs nothing installed to produce a
bundle with one. `scripts/make-icons.sh` re-derives the committed intermediates
when the artwork changes and is the only part that wants ImageMagick. The bundle
carries no weights.

`bundle-app.sh` signs with the first Developer ID in the keychain, under the
hardened runtime, and falls back to ad-hoc when there is none. Ad-hoc launches
on the machine that built it and nowhere else, so it is for development only.
`package-mac.sh` takes a Developer ID bundle the rest of the way: a disk image
with a drag target, signed, notarized and stapled. Stapling is the step that is
easy to skip and the one that decides whether a Mac offline at first launch has
to ask Apple.

Notarization credentials live in a keychain profile, made once and never read by
anything here:

```bash
xcrun notarytool store-credentials stemd-notary \
    --apple-id <apple id> --team-id <team id>
```

`STEMD_NOTARY_PROFILE` names a different one. Without a profile `package-mac.sh`
still produces a signed image and stops with what is missing, so an unnotarized
release takes a deliberate act.

```bash
STEMD_LINK_MODELS=1 ./scripts/bundle-app.sh    # symlink local weights
STEMD_EMBED_MODELS=1 ./scripts/bundle-app.sh   # copy weights in, offline install
```

## Flags

Three settings live in the window rather than here and are remembered between
launches: the model, and the output format and sample rate a client gets when it
does not ask for one. The flags below override them for one run without changing
what is saved, and the window greys out a control a flag has taken over.

`settings.json` in the user data directory holds them. It is plain JSON and safe
to edit or delete. Anything unreadable in it falls back to the default for that
field, and a saved model that will not load falls back rather than failing the
launch.

| flag | default | |
| --- | --- | --- |
| `--bind` | `0.0.0.0:8420` | |
| `--demucs-model` | saved, else `htdemucs` | any artefact in `--models` |
| `--output-format` | saved, else `flac` | `wav`, `mp3`, `pcm16`, `pcm32` |
| `--output-sample-rate` | saved, else `44100` | `24000`, `48000`, `96000` (not `mp3`) |
| `--settings` | user data directory | |
| `--overlap` | `0.25` | segment overlap fraction |
| `--full-precision` | off | float32 model, 1.3x slower |
| `--cache-dir` | user cache directory | emptied at every start |
| `--cache-max-gb` | `4` | about 80 tracks as flac, 37 as pcm16 |
| `--unfetched-ttl` | `300` | seconds |
| `--max-track-minutes` | `10` | the only size limit, body cap derived |
| `--queue-depth` | `16` | `429` past this |
| `--models` | `models` | searched before the download cache |
| `--instance` | `stemd` | mDNS name |
| `--no-mdns`, `--offline`, `--headless` | off | |

MP3's sample rates stop at 48 kHz: MPEG-1 Layer III has three and 96 is not one
of them, so that pair is refused rather than quietly encoded at 48.

`RUST_LOG` sets the level for both the console and the window's log. Without it
the console shows `info` and the window's buffer keeps `debug`, which its level
dropdown can reveal.

## Layout

```
crates/stemd-mlx/      demucs v4 on MLX: stft, layers, transformer, segmenting
crates/stemd-core/     pcm, stem topology, progress, the backend around the model
crates/stemd-audio/    file decoding, kept free of the model for the CLI's sake
crates/stemd-server/   axum API, cache, queue, mDNS, drops, window
crates/stemd-cli/      reference client: decode, upload, poll, rebuild, write
scripts/               bundling and packaging, one per platform
resources/             icon artwork, and what the build derives from it
tools/export/          PyTorch checkpoint to safetensors, run once per model
tools/eval/            fixture generators and the MUSDB benchmark
```

`stemd-mlx` is deliberately free of `stemd-core`: it is a model, and keeping it
standalone means its tests need nothing but weights. Every stage of it is nulled
against the reference implementation, and a whole track lands at -123.7 dB on
Metal.

That figure names a backend as much as it names the port. Two implementations of
the same arithmetic agree to their own floor rather than to a universal one, so
it is quoted for the backend that ships and would have to be re-derived stage by
stage for any other. `tests/model.rs` asserts -60 dB, which is the bar a port has
to clear. The -123.7 is what this one happens to reach.

`pyproject.toml` still exists and still lists torch. Nothing in it builds or runs
stemd. It is there to open published PyTorch checkpoints when a new model is
being converted, and to score models against MUSDB.

The repo tracks everything a clone needs to build and run. Gitignored: `models/`,
fetched on first run, `dist/` and `target/`. `tools/` ships nothing but is
tracked, because it is how the fixtures every null test reads were produced.

## Models

All three are served from
[this project's own release](https://github.com/nsaintot/stemd/releases/tag/models-v2)
and pinned by SHA-256. The demucs ones are
[mlx-community's](https://huggingface.co/mlx-community/demucs-mlx) conversions
byte for byte, mirrored so the app depends on one release rather than on an
upstream that could move. Nobody publishes an MLX BS-RoFormer, so that one is
converted here by `tools/export/convert_roformer.py`.

No manifest travels with any of them: the architecture is compiled in and every
layer checks the shape of the tensor it pulls, which catches the wrong artefact
more precisely than a JSON file claiming otherwise.

See [evaluation.md](evaluation.md) for why each preset is what it is.
