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

**macOS** 14 or later, Apple silicon, Metal. The Xcode command line tools,
`brew install cmake`, and the Metal shader compiler. About four minutes, which
is what those are for.

`metal` is the one that catches people out. MLX compiles its kernels during the
build, and the compiler no longer comes with the command line tools: it is a
separate component installed under an Xcode, so a machine can carry Xcode and a
current SDK and still not have it. `xcrun --find metal` says whether this one
does, and

```sh
xcodebuild -downloadComponent MetalToolchain
```

installs it if not. `scripts/bundle-app.sh` looks for it before building and
says this if it cannot find one.

The bundle is built against macOS 14 rather than against whatever SDK is to
hand. That is not tidiness: MLX bakes its Metal shaders in at build time and
does not compile them at runtime, so a floor taken from the build machine ships
an artefact that launches on an older Mac and then fails the first time it
wants the GPU. `MACOS_MIN` in `scripts/bundle-app.sh` and
`MACOS_DEPLOYMENT_TARGET` in `vendor/mlx-rs-stemd/mlx-sys/build.rs` are the two
halves of saying it, and they have to agree. The script passes its half
as a linker flag under `--target`; exported as `MACOSX_DEPLOYMENT_TARGET` it
would also reach proc-macro dylibs, which this toolchain links in a form dyld
refuses.

The kernels are a separate `mlx.metallib`, looked up beside the executable and
then at the build tree's absolute path. The bundle ships it in `Resources` with
a symlink from `MacOS`; without it the app only runs on the machine that built
it.

**Windows**, x86-64, CUDA or CPU. MSVC, the CUDA toolkit, cuDNN, and LLVM for
`libclang`. Use `scripts/bundle-windows.ps1` rather than setting the
environment by hand: it sets `MLX_BUILD_CUDA`, finds cuDNN, and then checks the
binary it produced actually carries the CUDA imports. That check exists because
a CPU-only build is silent. It links, starts, reports `device: cpu`, separates
correctly, and takes 610 seconds over a clip the card does in three.

Keep those variables identical between builds. cargo re-runs the MLX build
script when any of them moves, and that is forty minutes rather than four.

**Linux**, x86_64 or aarch64, CUDA. Build in the image from
`packaging/linux/Dockerfile`: CUDA 13.3 with cuDNN on Ubuntu 22.04, libclang
18, CMake 3.31, and the BLAS and avahi-compat headers. It needs no GPU.

```bash
docker build --platform linux/arm64 -t stemd-linux packaging/linux
docker run --rm -v "$PWD:/work" -e MLX_CUDA_ARCHITECTURES=121 \
    -e CMAKE_BUILD_PARALLEL_LEVEL=4 -e CARGO_BUILD_JOBS=4 stemd-linux \
    'cargo build --release -p stemd-server -p stemd-cli'
```

nvcc wants several GB per kernel, so the job count is capped: unbounded, it
runs a 32 GB machine out of memory. cmake-rs passes cargo's job count to the
build, so both are set.

On a machine of your own instead: `libavahi-compat-libdnssd-dev`,
`liblapacke-dev`, the CUDA toolkit, cuDNN 9.5 or later in `CUDNN_PATH`, and
`MLX_BUILD_CUDA=1`. cuDNN must be 9.5 or later: the cudnn-frontend MLX builds
against calls `cudnnBackendPopulateCudaGraph`, which was added there. An older
one compiles and then finds no execution plan for any graph at run time.

## Packaging

```bash
./scripts/bundle-app.sh          # macOS: dist/stemd.app, about 29 MB
./scripts/bundle.sh              # macOS: a signed, notarized, stapled .dmg
./scripts/bundle-windows.ps1     # Windows: dist/stemd-windows
./scripts/package-windows.ps1    # Windows: an installer and a zip
packaging/linux/stage.sh         # Linux: dist/linux, in the image
packaging/linux/build.sh         # Linux: .deb, .rpm and .AppImage from it
```

`MLX_CUDA_ARCHITECTURES` decides which cards the result runs on, and it is the
single largest thing in the size and the build time. Unset, MLX builds for the
card in the machine, which is a package that will not start anywhere else. The
released artefacts use the lists in `packaging/mlx-prebuilt.env`; five
architectures take about two and a half hours on sixteen cores.

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

### Release builds

`.github/workflows/cd.yml` builds every artefact on GitHub's runners and checks
that each installs and starts. A `v<version>` tag on main uploads them all to a
draft release, which is published by hand.

| | runner | check |
| --- | --- | --- |
| Linux `.deb`, `.rpm`, `.AppImage`, x86_64 and aarch64 | `ubuntu-24.04`, `ubuntu-24.04-arm` | installed on Ubuntu 22.04, Debian 12, Ubuntu 24.04 and Fedora |
| macOS `.dmg`, arm64 | `macos-26` | signed, notarized, `--version` |
| Windows installer and `.zip`, x86_64 | `windows-2022` | silent install, `--version` without CUDA |

The runners have no GPU, so a separation on a card is still checked by hand
before publishing.

macOS signing reads these repository secrets:

| secret | |
| --- | --- |
| `MACOS_CERT_P12` | base64 of the Developer ID Application certificate and key, as `.p12` |
| `MACOS_CERT_PASSWORD` | its password |
| `NOTARY_KEY_P8` | base64 of an App Store Connect API key (`.p8`), Developer role |
| `NOTARY_KEY_ID`, `NOTARY_ISSUER` | that key's ID, and for a team key its issuer ID (a UUID) |

Without them a pull request builds an ad hoc signed app and no `.dmg`, and a
release tag fails.

Windows installs CUDA 13.3.1 and cuDNN 9.14 from NVIDIA's redistributable
archives with `.github/scripts/windows-cuda.ps1`, the versions stemd fetches
at run time.

### Linux packages

| | |
| --- | --- |
| `stage.sh` | `dist/linux/opt/stemd`: binaries, vendored libraries, the run-time headers |
| `build.sh` | nfpm writes the `.deb` and `.rpm`, appimagetool the `.AppImage`, from that one tree |
| `nfpm.yaml.in` | dependencies: the display stack and D-Bus only |
| `THIRD-PARTY` | notices for the vendored NVIDIA, BLAS and avahi libraries |
| `jit-headers.sh` | writes `jit-headers.txt`, the toolkit headers MLX's run-time compiler reads; rerun after an MLX or CUDA update |

Every package carries CUDA, cuDNN, BLAS and avahi-compat in `/opt/stemd/lib`,
about 2 GB unpacked. The driver (`libcuda.so.1`), the display stack and the C++
runtime come from the machine. NVIDIA's libraries are copied unmodified, as
their licences require; the executables find them through `DT_RPATH`
(`$ORIGIN/../lib`), which also covers what cuDNN opens at run time.

`/usr/bin/stemd-server` is a symlink to `/opt/stemd/stemd`, a launcher that sets
`CUDA_HOME` to the bundled headers and execs `bin/stemd-server` by absolute
path. MLX locates its CCCL headers from argv[0] without resolving symlinks, so
the server must never be started through a symlink or by name.

The glibc floor is the image's: 2.35.

### Prebuilt MLX

Compiling MLX's CUDA kernels is most of a Linux or Windows build: about two and
a half hours on sixteen cores for five GPU architectures. The packages link a
prebuilt copy instead.

| | |
| --- | --- |
| `packaging/mlx-prebuilt.env` | release tag, GPU architectures per platform, SHA-256 per asset |
| `packaging/prebuilt.sh` | `build OUT` packs one, `fetch DIR` downloads and checks one; Linux in the image, Windows in Git Bash |
| `.github/workflows/mlx-prebuilt.yml` | builds all three on push of the `mlx-cuda-<n>` tag |
| `MLX_PREBUILT_OUT` | `mlx-sys` copies the MLX libraries and a manifest there after a build |
| `MLX_PREBUILT_DIR` | `mlx-sys` links from there and skips CMake |

On Windows the prebuilt also carries OpenBLAS (`libopenblas.lib` and `.dll`)
and `dl.lib`.

The manifest records the `mlx-sys` version, the MLX tag, the target, the C++
compiler (only `msvc` on Windows, whose toolsets link older ones), nvcc,
cuDNN, the GPU architectures and the RMSNorm patch, and `mlx-sys` refuses a
prebuilt that does not match its own build. To make a new one: bump `MLX_PREBUILT_TAG`, push that
tag, and copy the SHA-256 lines from the run summary into
`mlx-prebuilt.env`. A runner can stop at its 6 h limit before the kernels are
done; the compile goes through ccache, so re-running the failed job continues.

GPU architectures: Linux and Windows x86_64 `80;86;89;90a;120` (Ampere
through Blackwell), Linux aarch64 `90a;100;121` (Grace Hopper, Grace
Blackwell, DGX Spark). `90a` rather
than `90`, because MLX builds its Hopper-only kernels for `90a` alone.
Windows skips the Hopper quantized matmul kernels
(`MLX_DISABLE_SM90A_KERNELS`), which MSVC cannot compile; Hopper uses the
sm80 ones there.

On the macOS bundle: the icon is built from `resources/` with `sips` and
`iconutil`, both part of macOS, so a clone needs nothing installed to produce a
bundle with one. `scripts/make-icons.sh` re-derives the committed intermediates
when the artwork changes and is the only part that wants ImageMagick. The bundle
carries no weights.

`bundle-app.sh` signs with the first Developer ID in the keychain, under the
hardened runtime, and falls back to ad-hoc when there is none. Ad-hoc launches
on the machine that built it and nowhere else, so it is for development only.
`bundle.sh` takes a Developer ID bundle the rest of the way: a disk image
with a drag target, signed, notarized and stapled. Its window is drawn by
`resources/stemd-dmg.png`, the 2x rendition of a 640 by 400 window with two
placeholder squares on it; the icon positions at the top of the script are
those squares' centres in points, and follow the artwork if it moves them.
The image is laid out by Finder itself on a writable copy, which is what
writes a `.DS_Store`, and then compressed. Stapling is the step that is
easy to skip and the one that decides whether a Mac offline at first launch has
to ask Apple.

Notarization credentials live in a keychain profile, made once and never read by
anything here:

```bash
xcrun notarytool store-credentials stemd-notary \
    --apple-id <apple id> --team-id <team id>
```

`STEMD_NOTARY_PROFILE` names a different one. CI uses an App Store Connect API
key instead: `STEMD_NOTARY_KEY` (the `.p8` file), `STEMD_NOTARY_KEY_ID` and
`STEMD_NOTARY_ISSUER`. Without a profile `bundle.sh`
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
