# Building, packaging and flags

This page is for contributors. The [README](../README.md) is for people who use
the program.

## Clone

```bash
git clone --recurse-submodules https://github.com/nsaintot/stemd
cargo build --release
```

The submodule is `vendor/mlx-rs-stemd`, a fork of MLX's Rust bindings that
builds on platforms other than Apple silicon. Without it, cargo cannot resolve MLX.
For a clone made without `--recurse-submodules`, run
`git submodule update --init`.
The newest published `mlx-rs` and `mlx-sys` (0.25.3 and 0.2.0) declare no CUDA
feature, so the build uses the fork.

Every platform needs `cmake` on `PATH` and Rust 1.89 or later. MLX builds from
source on the first `cargo build`.

## Per platform

**macOS** 14 or later, Apple silicon, Metal. The build needs the Xcode command
line tools, `brew install cmake` and the Metal shader compiler, and takes about
four minutes.

The Metal shader compiler is the requirement most often missing. MLX compiles
its kernels during the build, and the compiler is not part of the command line
tools: it is a separate Xcode component, so a machine
can have Xcode and a current SDK and still lack it. `xcrun --find metal` shows
whether it is installed, and

```sh
xcodebuild -downloadComponent MetalToolchain
```

installs it. `scripts/bundle-app.sh` checks for the compiler before building and
prints this command if it cannot find one.

The bundle is built for macOS 14, whatever the SDK on the build machine. MLX
compiles its Metal shaders at build time and never at run time, so a bundle
built for a newer macOS launches on an older Mac and then fails the first time
it uses the GPU. `MACOS_MIN` in `scripts/bundle-app.sh` and
`MACOS_DEPLOYMENT_TARGET` in `vendor/mlx-rs-stemd/mlx-sys/build.rs` set this
floor and must agree. The script passes `MACOS_MIN` to the linker in `RUSTFLAGS` and builds with
`--target`, so only the executable gets the flag. Do not export it as `MACOSX_DEPLOYMENT_TARGET`: that also reaches
proc-macro dylibs, which this toolchain then links in a form that dyld refuses.

The kernels are in a separate `mlx.metallib`, looked up next to the executable
and then at the build tree's absolute path. The bundle ships it in `Resources`
with a symlink from `MacOS`; without it, the app runs only on the machine that
built it.

**Windows**, x86-64, CUDA or CPU. The build needs MSVC, the CUDA toolkit, cuDNN,
and LLVM for `libclang`. Use `scripts/bundle-windows.ps1` instead of setting the
environment by hand: it sets `MLX_BUILD_CUDA`, finds cuDNN, and checks that the
binary it produced carries the CUDA imports. A CPU-only build gives no sign of
the problem: it links, starts, reports `device: cpu`, separates correctly, and
takes 610 seconds on a 64-second clip that the GPU separates in a few seconds.

Keep those variables identical between builds. cargo re-runs the MLX build
script when any of them changes, which takes forty minutes instead of four.

**Linux**, x86_64 or aarch64, CUDA. Build in the image from
`packaging/linux/Dockerfile`, which has CUDA 13.3 with cuDNN on Ubuntu 22.04,
libclang 18, CMake 3.31, and the BLAS and avahi-compat headers. The build needs
no GPU.

```bash
docker build --platform linux/arm64 -t stemd-linux packaging/linux
docker run --rm -v "$PWD:/work" -e MLX_CUDA_ARCHITECTURES=121 \
    -e CMAKE_BUILD_PARALLEL_LEVEL=4 -e CARGO_BUILD_JOBS=4 stemd-linux \
    'cargo build --release -p stemd-server -p stemd-cli'
```

nvcc needs several GB of memory per kernel, so the job count is capped: without
a cap, the build runs a 32 GB machine out of memory. cmake-rs passes cargo's job
count to the build, so the command sets both.

To build on your own machine instead, install `libavahi-compat-libdnssd-dev`,
`liblapacke-dev`, the CUDA toolkit and cuDNN 9.5 or later, set `CUDNN_PATH` to
cuDNN, and set `MLX_BUILD_CUDA=1`. The cudnn-frontend that MLX builds against
calls `cudnnBackendPopulateCudaGraph`, which first appeared in cuDNN 9.5. An
older cuDNN builds, but finds no execution plan at run time.

## Packaging

```bash
./scripts/bundle-app.sh          # macOS: dist/Stemd.app, about 29 MB
./scripts/bundle.sh              # macOS: a signed, notarized, stapled .dmg
./scripts/bundle-windows.ps1     # Windows: dist/stemd-windows
./scripts/package-windows.ps1    # Windows: an installer and a zip
packaging/linux/stage.sh         # Linux: dist/linux, in the image
packaging/linux/build.sh         # Linux: .deb, .rpm and .AppImage from it
```

`MLX_CUDA_ARCHITECTURES` sets which GPUs the result runs on, and it has the
largest effect on package size and build time. When it is unset, MLX builds only
for the GPU in the build machine, and the package starts nowhere else. The
released artefacts use the lists in `packaging/mlx-prebuilt.env`; five
architectures take about two and a half hours on sixteen cores.

The Windows installer offers two modes. The default, Just for me, installs into
`%LOCALAPPDATA%\Programs\Stemd`; the all-users mode installs into Program Files.
The per-user install goes under `Programs` because `%LOCALAPPDATA%\stemd` holds
the program's weights and cache, and an uninstaller pointed at that directory
would delete a gigabyte of downloads.

The two modes differ in how they install CUDA. `install-cuda.cmd` writes about
1.2 GB next to the executable, which a per-user install can do and a Program
Files install cannot. The all-users install therefore replaces that script with
one that asks for administrator rights, so it does not fail on a permission
error after a gigabyte of download.

Elevation happens at launch: stock NSIS cannot raise its own token during a run,
and the plugin that can is not part of NSIS. The installer manifest asks for the
highest token available, which on an administrator account means one consent
prompt even for a per-user install. Building the installer needs NSIS, which
`winget install NSIS.NSIS` provides.

### Notices

Every package ships the files listed in `packaging/notices.list`: the two stemd
licences, the notices and the licence texts. `bundle-app.sh` copies them into
`Stemd.app/Contents/Resources`, `package-windows.ps1` beside the executables,
and `stage.sh` into `/opt/stemd/share/doc/stemd`.

| file | |
| --- | --- |
| `packaging/THIRD-PARTY` | every bundled component, its licence, and the platforms that ship it |
| `packaging/THIRD-PARTY-CRATES` | the Rust crates and their licence texts, generated |
| `packaging/licenses/` | licence texts the repository has no other copy of |
| `packaging/about.toml`, `packaging/about.hbs` | the cargo-about config and template |
| `scripts/crate-notices.sh` | regenerates `THIRD-PARTY-CRATES` |

Rerun `scripts/crate-notices.sh` after a dependency change and commit the
result. It needs cargo-about 0.9.3:

```bash
cargo install cargo-about --locked --version 0.9.3 --features cli
```

Edit `packaging/THIRD-PARTY` and `packaging/licenses/MLX.txt` by hand after an
MLX, CUDA or Sparkle update.

### Release builds

`.github/workflows/cd.yml` builds every artefact on GitHub's runners and checks
that each one installs and starts. A `v<version>` tag on main uploads them all
to a draft release, which is published by hand.

| | runner | check |
| --- | --- | --- |
| Linux `.deb`, `.rpm`, `.AppImage`, x86_64 and aarch64 | `ubuntu-24.04`, `ubuntu-24.04-arm` | installed on Ubuntu 22.04, Debian 12, Ubuntu 24.04 and Fedora |
| macOS `.dmg`, arm64 | `macos-26` | signed, notarized, `--version` |
| Windows installer and `.zip`, x86_64 | `windows-2022` | silent install, `--version` without CUDA |

The runners have no GPU, so a GPU separation is checked by hand before
publishing.

macOS signing reads these repository secrets:

| secret | |
| --- | --- |
| `MACOS_CERT_P12` | base64 of the Developer ID Application certificate and key, as `.p12` |
| `MACOS_CERT_PASSWORD` | the password of the `.p12` |
| `NOTARY_KEY_P8` | base64 of an App Store Connect API key (`.p8`), Developer role |
| `NOTARY_KEY_ID`, `NOTARY_ISSUER` | that key's ID, and for a team key its issuer ID (a UUID) |

Without these secrets, a pull request builds an ad hoc signed app and no `.dmg`,
and a release tag fails.

A release is an annotated tag whose message holds the release notes for users,
in Markdown, with `## Added`, `## Modified` and `## Fixed` sections:

```bash
git tag -a --cleanup=whitespace -F notes.md v0.2.0 origin/main
git push origin v0.2.0
```

`--cleanup=whitespace` keeps the Markdown headings, because the default cleanup
drops every line that starts with `#`.

`.github/scripts/release-draft.sh` fills the draft:

| part of the draft | content |
| --- | --- |
| text | the notes, a Downloads table with a link per package, and GitHub's generated list of pull requests in a collapsed block |
| `appcast.xml` | the macOS app's Sparkle feed, with the notes as HTML |
| `SHA256SUMS.txt` | one line per package |

With a lightweight tag or an empty message, the notes are GitHub's generated
list. To correct the notes of a draft without a new build:

```bash
gh workflow run release-notes.yml -f tag=v0.2.0 -F notes=@notes.md
```

Neither CD nor `release-notes.yml` changes a published release. The app finds
the feed under GitHub's "latest" release, so publish model releases
(`models-v*`) with `gh release create --latest=false`.

| secret | |
| --- | --- |
| `UPDATE_SIGNING_KEY` | base64 of the 32-byte Ed25519 seed that signs the `.dmg` in the feed; the public half is `packaging/update-ed25519.pub` |

The Windows runner installs CUDA 13.3.1 and cuDNN 9.14 from NVIDIA's
redistributable archives with `.github/scripts/windows-cuda.ps1`. These are the
versions stemd fetches at run time.

### Linux packages

| | |
| --- | --- |
| `stage.sh` | `dist/linux/opt/stemd`: binaries, vendored libraries, the run-time headers |
| `build.sh` | nfpm writes the `.deb` and `.rpm`, appimagetool the `.AppImage`, from that one tree |
| `nfpm.yaml.in` | the package dependencies, which are only the display stack and D-Bus |
| `jit-headers.sh` | writes `jit-headers.txt`, the list of toolkit headers that MLX's run-time compiler reads; rerun it after an MLX or CUDA update |

Every package carries CUDA, cuDNN, BLAS and avahi-compat in `/opt/stemd/lib`,
about 1.7 GB unpacked. The driver (`libcuda.so.1`), the display stack and the
C++ runtime come from the machine. NVIDIA's libraries are copied unmodified, as
their licences require. The executables find them through `DT_RPATH`
(`$ORIGIN/../lib`), which also covers the libraries cuDNN opens at run time.

`/usr/bin/stemd-server` is a symlink to `/opt/stemd/stemd`, a launcher that sets
`CUDA_HOME` to the bundled headers and execs `bin/stemd-server` by absolute
path. MLX locates its CCCL headers from argv[0] without resolving symlinks, so
never start the server binary through a symlink or by name.

The minimum glibc is 2.35, from the build image.

### Prebuilt MLX

Compiling MLX's CUDA kernels takes most of a Linux or Windows build: about two
and a half hours on sixteen cores for five GPU architectures. The packages link
a prebuilt copy of MLX instead.

| | |
| --- | --- |
| `packaging/mlx-prebuilt.env` | the release tag, the GPU architectures per platform and the SHA-256 of each asset |
| `packaging/prebuilt.sh` | `build OUT` packs a prebuilt and `fetch DIR` downloads and checks one; it runs in the image on Linux and in Git Bash on Windows |
| `.github/workflows/mlx-prebuilt.yml` | builds all three prebuilts when an `mlx-cuda-<n>` tag is pushed |
| `MLX_PREBUILT_OUT` | `mlx-sys` copies the MLX libraries and a manifest there after a build |
| `MLX_PREBUILT_DIR` | `mlx-sys` links from there and skips CMake |

On Windows, the prebuilt also carries OpenBLAS (`libopenblas.lib` and `.dll`)
and `dl.lib`.

`mlx-sys` patches MLX's CUDA allocator so that a CUDA build also runs on the CPU
of a machine without a CUDA device. CD checks this with a CPU separation on the
Linux and Windows runners, which have no GPU.

The manifest records the `mlx-sys` version, the MLX tag, the target, the C++
compiler, nvcc, cuDNN, the GPU architectures and the RMSNorm patch. On Windows
it records only `msvc` as the compiler, because newer MSVC toolsets link
libraries built by older ones. `mlx-sys` refuses a prebuilt that does not match
its own build.

To make a new prebuilt, bump `MLX_PREBUILT_TAG`, push that tag, and copy the
SHA-256 lines from the run summary into `mlx-prebuilt.env`. A runner can hit its
6 h limit before the kernels are compiled. The compile goes through ccache, so
re-running the failed job picks up where it stopped.

GPU architectures: Linux and Windows x86_64 use `80;86;89;90a;120` (Ampere
through Blackwell), and Linux aarch64 uses `90a;100;121` (Grace Hopper, Grace
Blackwell, DGX Spark). The list has `90a` in place of `90` because MLX builds
its Hopper-only kernels for `90a` alone. Windows skips the Hopper quantized
matmul kernels (`MLX_DISABLE_SM90A_KERNELS`), which MSVC cannot compile, so
Hopper GPUs use the sm80 kernels there.

### macOS bundle

The icon is built from `resources/` with `sips` and `iconutil`, both part of
macOS, so a clone needs nothing extra installed to produce a bundle with an
icon. `scripts/make-icons.sh` regenerates the committed intermediate images when
the artwork changes, and it is the only step that needs ImageMagick. The bundle
carries no weights.

`bundle-app.sh` signs with the first Developer ID in the keychain, under the
hardened runtime, and falls back to ad-hoc signing when there is none. An ad-hoc
bundle launches only on the machine that built it, so it is for development
only.

`bundle.sh` turns a Developer ID bundle into a disk image with a drag target,
then signs, notarizes and staples it. The disk image window is drawn by
`resources/stemd-dmg.png`, the 2x rendition of a 640 by 400 window with two
placeholder squares on it. The icon positions at the top of the script are the
centres of those squares in points; update them if the artwork moves the
squares. Finder itself lays out the image on a writable copy, which writes the
`.DS_Store`, and the script then compresses it. Do not skip stapling. Without it,
a Mac offline at first launch cannot check the notarization.

The notarization credentials live in a keychain profile, created once. Nothing
in this repository reads it:

```bash
xcrun notarytool store-credentials stemd-notary \
    --apple-id <apple id> --team-id <team id>
```

`STEMD_NOTARY_PROFILE` names a different profile. CI uses an App Store Connect
API key instead: `STEMD_NOTARY_KEY` (the `.p8` file), `STEMD_NOTARY_KEY_ID` and
`STEMD_NOTARY_ISSUER`. Without a profile, `bundle.sh` still produces a signed
image and then stops and names what is missing, so an unnotarized release cannot
happen by accident.

```bash
STEMD_LINK_MODELS=1 ./scripts/bundle-app.sh    # symlink local weights
STEMD_EMBED_MODELS=1 ./scripts/bundle-app.sh   # copy weights in, offline install
```

## Flags

The window sets some settings and remembers them between launches: the model,
the output format and sample rate that a client gets when it does not ask for
one, and, in the Settings sheet, the longest track, the port, and the cache size
and age. The flags below override these settings for one run without changing
what is saved, and the window greys out a control that a flag overrides.

The settings are saved in `settings.json` in the user data directory. It is
plain JSON and safe to edit or delete. An unreadable field falls back to the
default for that field, and a saved model that does not load falls back to the
default model instead of failing the launch.

| flag | default | |
| --- | --- | --- |
| `--bind` | `0.0.0.0` on the saved port | when the saved port is in use, the server falls back to `8420`, then to any free port; when the `--bind` address is in use, the launch fails |
| `--demucs-model` | saved, else `htdemucs` | any artefact in `--models` |
| `--output-format` | saved, else `flac` | also `wav`, `mp3`, `pcm16`, `pcm32` |
| `--output-sample-rate` | saved, else `44100` | also `24000`, `48000`, `96000` (not with `mp3`) |
| `--settings` | user data directory | |
| `--overlap` | `0.25` | the fraction by which segments overlap |
| `--full-precision` | off | runs every model at float32. By default Metal runs every model at float16 (1.3x faster), CUDA runs demucs at float16 and the RoFormer at float32, and the CPU runs float32 |
| `--cache-dir` | user cache directory | kept across restarts; the alias is `--pieces-dir` |
| `--cache-max-gb` | saved, else `8` | 8 GB holds about 200 six-minute tracks as 44.1 kHz flac; the alias is `--pieces-max-gb` |
| `--partial-max-hours` | `24` | how long a track with a stem still in pieces is kept unused; `0` keeps it until the cache is full |
| `--whole-max-days` | saved, else `30` | how long a track with every stem whole is kept unused; `0` keeps it until the cache is full |
| `--ephemeral` | off | empties the cache at start and at quit |
| `--unfetched-ttl` | `300` | seconds a job result is held for its client |
| `--max-track-minutes` | saved, else `10` | the only size limit; the body cap is derived from it. Memory caps it, and `0` takes that cap |
| `--queue-depth` | `16` | jobs past this many get `429` |
| `--session-idle-secs` | `30` | a session with nothing to run and no request for this long is ended |
| `--max-sessions` | `4` | sessions open at once; each holds its track in memory |
| `--models` | `models` | searched before the download cache |
| `--instance` | `stemd` | the mDNS name |
| `--no-mdns`, `--offline`, `--headless` | off | |
| `--install-cuda` | off | Windows only: downloads the CUDA runtime beside the executable and exits |

MP3 sample rates stop at 48 kHz: MPEG-1 Layer III has three rates, and 96 kHz is
not one of them. The server refuses `mp3` at 96 kHz instead of encoding it at 48
kHz.

`RUST_LOG` sets the level for both the console and the window's log. Without it,
the console shows `info` and the window's buffer keeps `debug`, which the
window's level dropdown can show.

## Layout

```
crates/stemd-mlx/      demucs v4 and BS-RoFormer on MLX, Metal or CUDA: stft, layers, transformer, segmenting
crates/stemd-core/     pcm, stem topology, progress, the backend around the model
crates/stemd-audio/    file decoding, with no model dependency so the CLI stays small
crates/stemd-server/   axum API, cache, queue, mDNS, drops, window
crates/stemd-cli/      reference client: decode, upload, poll, rebuild, write
scripts/               bundling and packaging, one per platform
resources/             icon artwork, and what the build derives from it
tools/export/          PyTorch checkpoint to safetensors, run once per model
tools/eval/            fixture generators and the MUSDB benchmark
```

`stemd-mlx` does not depend on `stemd-core`: it is a model, and as a standalone
crate its tests need nothing but weights. Every stage of it is nulled against
the reference implementation, and a whole track nulls at -123.7 dB on Metal.

That figure holds for Metal only. Two implementations of the same arithmetic
agree down to the precision floor of their backend, so another backend must
measure its own figures stage by stage. `tests/model.rs` asserts -60 dB, the bar
every port must clear.

`pyproject.toml` still lists torch. Nothing in it builds or runs stemd: it opens
published PyTorch checkpoints when a new model is converted, and scores models
against MUSDB.

The repository tracks everything a clone needs to build and run. `models/`
(downloaded on first run), `dist/` and `target/` are gitignored. `tools/` ships
nothing but is tracked, because it produced the fixtures that every null test
reads.

## Models

The models are served from this project's own releases and pinned by SHA-256:
the demucs models from
[models-v2](https://github.com/nsaintot/stemd/releases/tag/models-v2) and
`stemd_roformer` from
[models-v3](https://github.com/nsaintot/stemd/releases/tag/models-v3). The
demucs models are byte-for-byte copies of
[mlx-community's](https://huggingface.co/mlx-community/demucs-mlx) conversions,
mirrored so the app depends only on this project's releases. Nobody publishes an
MLX BS-RoFormer, so `tools/export/convert_stemd_roformer.py` and then
`tools/export/convert_roformer.py` convert that one here. See
[models.md](models.md) for the files and digests.

No manifest comes with the weights. The architecture is compiled in, and every
layer checks the shape of each tensor it reads, which identifies a wrong
artefact more precisely than a manifest file could.

See [evaluation.md](evaluation.md) for why each preset is what it is.
