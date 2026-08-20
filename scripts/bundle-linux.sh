#!/bin/bash
# Build and package stemd for Linux, with CUDA if the machine has it.
#
# The counterpart to bundle-windows.ps1, and it exists for the same reason that
# one now sets its own environment. MLX builds its CUDA backend only when
# MLX_BUILD_CUDA says so, and says nothing at all when it does not: the result
# links, starts, reports `device: cpu`, separates correctly, and runs about 190x
# slower. That was measured, not guessed. A build that depends on the variable
# already being exported is a build that works in one shell.
#
# Everything it needs from the machine is checked before anything is compiled,
# because the failures are otherwise sixty lines into a CMake trace and name
# things that are not what is missing.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-$root/dist/stemd-linux}"

# The architecture of the card this is for. One keeps the compile short; a
# release wants a wider list and a much longer build. sm_86 is the RTX 3090 Ti
# this was developed against.
export MLX_CUDA_ARCHITECTURES="${MLX_CUDA_ARCHITECTURES:-86}"
# nvcc's front end wants several GB per qmm_impl_* kernel and there are 107 of
# them. Unbounded, it took a 32 GB Windows machine and a 16 GB WSL out of memory
# at about two thirds of the way through. Raise it if the machine is large.
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"

say() { printf '  %s\n' "$*"; }

# The cuDNN version cudnn-frontend needs. See the search below.
CUDNN_MIN=9.5

# Read a cuDNN version out of a directory holding cudnn.h. The numbers moved
# into cudnn_version.h at 8.0, and Debian puts that one in the multiarch
# directory rather than beside the header that includes it, so both are tried.
cudnn_version() {
    local inc="$1" f v
    for f in "$inc/cudnn_version.h" "$inc"/*/cudnn_version.h "$inc/cudnn.h"; do
        [ -f "$f" ] || continue
        v="$(awk '/define CUDNN_MAJOR/{a=$3} /define CUDNN_MINOR/{b=$3} \
                  /define CUDNN_PATCHLEVEL/{c=$3} \
                  END{if (a != "") print a"."b"."c}' "$f")"
        [ -n "$v" ] && { printf '%s' "$v"; return; }
    done
}
die() { printf '\nstopped: %s\n' "$*" >&2; exit 1; }

# --- what the machine has ------------------------------------------------
#
# Seven things had to be true for this to build the first time and not one of
# them is in any README, so they are checked here by name rather than left to
# fail as something else. See docs in the port notes for what each one costs.

# cargo, which is the one thing here that is not about CUDA and the one this
# script kept assuming. rustup installs it into ~/.cargo/bin and adds that to
# PATH from a shell profile, which a non-interactive ssh session never reads.
# So this ran fine by hand and died with `cargo: command not found` when run
# over ssh, after passing every other check and printing all of them.
if ! command -v cargo >/dev/null 2>&1; then
    for c in "${CARGO_HOME:-$HOME/.cargo}/bin" /usr/local/cargo/bin; do
        [ -x "$c/cargo" ] && export PATH="$c:$PATH" && break
    done
fi
command -v cargo >/dev/null 2>&1 ||
    die "no cargo on PATH, and none under ~/.cargo/bin. If rustup installed it
        somewhere else, put that directory on PATH; a login shell finding it is
        not enough, because this may be running without one."
say "cargo: $(command -v cargo) ($(cargo --version | awk '{print $2}'))"

# Two layouts in the wild and they agree on almost nothing. NVIDIA's run-file
# and repository packages put everything under /usr/local/cuda-<version>, nvcc
# included. Debian packages the toolkit as /usr/lib/cuda and puts nvcc in
# /usr/bin, so testing for bin/nvcc under the root finds nothing on a machine
# that has a perfectly good toolkit.
cuda_home="${CUDA_PATH:-${CUDA_HOME:-}}"
if [ -z "$cuda_home" ]; then
    cuda_home="$(ls -d /usr/local/cuda-* /usr/local/cuda /usr/lib/cuda 2>/dev/null | sort -Vr | head -1 || true)"
fi
# The chosen toolkit's own nvcc wins over whatever is on PATH. A machine with
# both Debian's 12.4 in /usr/bin and NVIDIA's 13.3 in /usr/local resolves
# `command -v nvcc` to the 12.4 one, and then reports building against 13.3
# while compiling with 12.4: the binary comes out linked to libcudart.so.12 and
# gets packaged beside 13.3's libraries and headers.
nvcc=""
[ -x "${cuda_home:-/nonexistent}/bin/nvcc" ] && nvcc="$cuda_home/bin/nvcc"
[ -z "$nvcc" ] && nvcc="$(command -v nvcc || true)"

if [ -n "$cuda_home" ] && [ -n "$nvcc" ]; then
    export CUDA_PATH="$cuda_home"
    export CUDA_HOME="$cuda_home"
    export MLX_BUILD_CUDA=1
    [ -d "$cuda_home/bin" ] && export PATH="$cuda_home/bin:$PATH"
    export CUDACXX="$nvcc"
    say "CUDA: $cuda_home, nvcc $("$nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)"
    say "building for sm_$MLX_CUDA_ARCHITECTURES, $CMAKE_BUILD_PARALLEL_LEVEL at a time"
    cuda=yes
else
    say "no CUDA toolkit found, so this will be a CPU-only build"
    say "set CUDA_PATH if there is one somewhere unusual"
    cuda=no
fi

# CMake 3.25, which is bookworm's, fails find_package on CUDA::nvToolsExt, a
# target CUDA 12 removed. Only matters for a CUDA build.
if [ "$cuda" = yes ]; then
    have="$(cmake --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)"
    [ -n "$have" ] || die "no cmake on PATH"
    if [ "$(printf '%s\n3.31\n' "$have" | sort -V | head -1)" != "3.31" ]; then
        die "cmake $have is too old: 3.25 fails find_package on CUDA::nvToolsExt,
        a target CUDA 12 removed. 3.31 or newer, and a build of it into /opt is
        the usual answer on Debian."
    fi
    say "cmake: $have"
fi

# MLX find_path()s for lapacke.h, so the headers have to be there and not just
# the libraries. liblapacke-dev on Debian.
[ -f /usr/include/lapacke.h ] || ls /usr/include/*/lapacke.h >/dev/null 2>&1 ||
    die "no lapacke.h: MLX looks for it by path, not just for the library.
        Install liblapacke-dev (and libopenblas-dev, liblapack-dev)."

# bindgen needs libclang, and Debian's clang 14 knows neither _Float16 nor
# __bf16 on x86, which is what mlx-c's half.h is made of.
if [ -z "${LIBCLANG_PATH:-}" ]; then
    for v in 19 18 17 16; do
        if [ -d "/usr/lib/llvm-$v/lib" ]; then export LIBCLANG_PATH="/usr/lib/llvm-$v/lib"; break; fi
    done
fi
[ -n "${LIBCLANG_PATH:-}" ] ||
    die "no libclang 16 or newer found for bindgen. Debian's clang 14 rejects
        _Float16 and __bf16 on x86, which mlx-c's half.h uses. Install
        libclang-16-dev or newer, or set LIBCLANG_PATH."
say "libclang: $LIBCLANG_PATH"

# The linker cache, read once. dpkg rewrites /etc/ld.so.cache from a trigger at
# the end of an install, and `ldconfig -p` run at that moment lists nothing, so
# a check that shells out separately each time occasionally reports a library
# missing that is sitting on the disk. That happened twice today, both times
# minutes after an apt run, and both times the library was there. Falling back
# to looking for the file makes the answer not depend on the timing.
libcache="$(/sbin/ldconfig -p 2>/dev/null || true)"
have_lib() {
    case "$libcache" in *"$1"*) return 0 ;; esac
    ls /usr/lib/*/"$1"* /lib/*/"$1"* /usr/lib/"$1"* 2>/dev/null | grep -q . 
}

# `-ldns_sd` is avahi-compat on Linux, and discovery.rs links it unconditionally.
have_lib libdns_sd ||
    die "no libdns_sd: mDNS advertisement links it. Install
        libavahi-compat-libdnssd-dev."

# `-lcuda` is the driver API. On a machine with the driver it is in the usual
# place; on one with only the toolkit it is the stub, and MLX links it either
# way.
if [ "$cuda" = yes ]; then
    have_lib libcuda.so ||
        [ -e "$cuda_home/lib64/stubs/libcuda.so" ] ||
        die "no libcuda.so to link against. Either install the NVIDIA driver, or
            symlink the toolkit's lib64/stubs/libcuda.so somewhere on the
            linker path."
fi

# MLX's CMakeLists does find_package(CUDNN REQUIRED) for the CUDA backend, and
# its FindCUDNN.cmake looks only in the toolkit and one or two NVIDIA paths.
# Absent, the build dies sixty lines into a CMake trace asking to be passed
# CUDNN_INCLUDE_PATH, which is not a variable anything here sets and not the one
# that would fix it. This cost a build on both platforms today.
#
# It is not part of the CUDA toolkit on either. Debian ships `nvidia-cudnn`,
# which is a downloader that shows NVIDIA's licence and asks you to accept it.
if [ "$cuda" = yes ]; then
    # Not first-match, and that matters. Debian ships cuDNN 9.0.0, while the
    # cudnn-frontend MLX pins calls cudnnBackendPopulateCudaGraph, which cuDNN
    # added in 9.5. Taking the first cudnn.h on a machine that has both gives a
    # build that compiles cleanly and then fails at run time with "No execution
    # plans support the graph", on a server that started and said device: gpu.
    # So every candidate is read for its version and the newest usable one wins.
    #
    # /opt is in the list because that is where a manual install lands, and on
    # Debian 13 a manual install is the only kind there is: NVIDIA publish cuDNN
    # for ubuntu2204, ubuntu2404 and debian12, and not for trixie.
    cudnn_root=""
    cudnn_ver=""
    cudnn_seen=""
    for cand in "${CUDNN_PATH:-}" /opt/cudnn* /opt/*/cudnn "$cuda_home" /usr /usr/local; do
        [ -n "$cand" ] && [ -f "$cand/include/cudnn.h" ] || continue
        v="$(cudnn_version "$cand/include")"
        [ -n "$v" ] || continue
        cudnn_seen="$cudnn_seen $cand ($v)"
        # Below the floor never wins, however new it is relative to the others.
        [ "$(printf '%s\n%s\n' "$v" "$CUDNN_MIN" | sort -V | head -1)" = "$CUDNN_MIN" ] || continue
        if [ -z "$cudnn_ver" ] ||
           [ "$(printf '%s\n%s\n' "$v" "$cudnn_ver" | sort -Vr | head -1)" = "$v" ]; then
            cudnn_root="$cand"
            cudnn_ver="$v"
        fi
    done
    if [ -z "$cudnn_root" ] && [ -n "$cudnn_seen" ]; then
        die "cuDNN is installed but too old:$cudnn_seen. MLX builds against a
            cudnn-frontend that calls cudnnBackendPopulateCudaGraph, which cuDNN
            added in $CUDNN_MIN. An older one links, starts, loads a model, and
            then finds no execution plan for any graph. Install $CUDNN_MIN or
            newer and set CUDNN_PATH to the root holding include/cudnn.h."
    fi
    [ -n "$cudnn_root" ] ||
        die "no cudnn.h anywhere this looked: CUDNN_PATH, /opt, $cuda_home, /usr.
            MLX needs cuDNN for the CUDA backend and it is not part of the CUDA
            toolkit. Where it comes from depends on the distribution: on Ubuntu
            and on Debian 12 it is apt install libcudnn9-cuda-13, and on Debian
            13 there is no package yet, so it has to be unpacked by hand and
            named with CUDNN_PATH."
    export CUDNN_PATH="$cudnn_root"
    say "cudnn: $cudnn_root ($cudnn_ver)"
fi

# MLX compiles some kernels at run time with NVRTC and they include
# <cuda/std/tuple>. cuda-toolkit does not pull CCCL in, and without it every
# separation dies on the first gather, at run time, on a machine that started
# fine and reported device: gpu.
cccl=""
if [ "$cuda" = yes ]; then
    # Whatever directory holds cuda/std/, since that is the include root MLX
    # hands NVRTC. The run-file groups cuda/, thrust/ and cub/ under a cccl
    # directory; Debian puts all three straight into /usr/include.
    for c in "$cuda_home/targets/x86_64-linux/include/cccl" "$cuda_home/include/cccl" \
             "$cuda_home/include" /usr/include; do
        [ -f "$c/cuda/std/tuple" ] && cccl="$c" && break
    done
    [ -n "$cccl" ] ||
        die "no CCCL headers under $cuda_home. MLX builds kernels at run time
            and they include <cuda/std/tuple>, so a machine that never compiles
            anything still needs them on disk. Install cuda-cccl-<version>."
    say "cccl: $cccl"
fi

# MLX compiles kernels at run time and needs the CUDA runtime headers as well
# as CCCL: jit_module.cpp looks for them under CUDA_HOME/include, and without
# them every separation fails with "Can not find locations of CUDA headers".
# The toolkit layout has them beside the toolkit; Debian puts them in
# /usr/include with everything else.
cuda_inc=""
if [ "$cuda" = yes ]; then
    for c in "$cuda_home/include" /usr/include; do
        [ -f "$c/cuda_runtime.h" ] && cuda_inc="$c" && break
    done
    [ -n "$cuda_inc" ] || die "no cuda_runtime.h under $cuda_home or /usr/include."
    say "cuda headers: $cuda_inc"
fi

# --- build ---------------------------------------------------------------

printf '\nbuilding...\n'
cargo build --release --manifest-path "$root/Cargo.toml" -p stemd-server -p stemd-cli

exe="$root/target/release/stemd-server"

# Ask the binary, not the environment that made it. On Linux these are ordinary
# link-time dependencies rather than Windows' delay-loads, so ldd is the whole
# answer and needs no tooling that might be absent.
if [ "$cuda" = yes ]; then
    linked="$(ldd "$exe" | grep -cE 'libcublas|libcufft|libcudnn|libnvrtc' || true)"
    [ "$linked" -ge 3 ] ||
        die "$exe links no CUDA libraries, so MLX_BUILD_CUDA did not take. It
            would ship as a server that starts, reports device: cpu, separates
            correctly and runs about 190x slower. Remove
            target/release/build/mlx-sys-* and build again."
    say "links $linked CUDA libraries"
fi

# --- package -------------------------------------------------------------
#
# bin/ rather than the top level, because MLX resolves its NVRTC include path
# as <binary_dir>/../include/cccl. Put the executable at the root and the
# headers are looked for one directory above the package.

rm -rf "$out"
mkdir -p "$out/bin"
cp "$root/target/release/stemd-server" "$root/target/release/stemd-cli" "$out/bin/"

if [ "$cuda" = yes ]; then
    mkdir -p "$out/lib" "$out/include"
    # What it actually links, resolved through ldd, plus the two families that
    # are loaded at run time and so never appear there: cuDNN opens its engine
    # libraries itself, and nvrtc wants its builtins the same way.
    # cudnn is deliberately not in this list. ldd resolves libcudnn.so.9 through
    # the system linker, which finds the distribution's copy, and cuDNN 9 splits
    # into a small dispatcher plus large engine libraries that it opens itself.
    # Taking the dispatcher from here and the engines from the cuDNN actually
    # built against gives a package with a 9.0 dispatcher driving 9.14 engines,
    # which loads, reports its version as 9.0, and then finds no execution plan
    # for any graph. All of cuDNN comes from one place, below.
    ldd "$exe" | awk '{print $1, $3}' |
        grep -E '^lib(cudart|cublas|cublasLt|cufft|nvrtc|nvJitLink|cusparse|curand)' |
        while read -r name path; do
            [ -f "$path" ] && cp -Lu "$path" "$out/lib/" && say "bundled $name"
        done
    # cuDNN opens its engine libraries itself and nvrtc wants its builtins the
    # same way, so ldd cannot see either. Taken from the cuDNN actually built
    # against rather than from a fixed path, which on a machine with two
    # installed would otherwise bundle the wrong one.
    #
    # The builtins come from beside the nvrtc that ldd resolved, for the same
    # reason and by the same rule. Globbing a system directory by name does not
    # keep the pair together: this machine has Debian's CUDA 12.4 in
    # /usr/lib/x86_64-linux-gnu and NVIDIA's 13.3 under /usr/local, and the glob
    # bundled 10.7 MB of 12.4 builtins next to the 13.3 nvrtc the binary links.
    # Harmless there, because nvrtc loads its own version by name, but it is
    # dead weight that says the sweep is picking files by what they are called
    # rather than by what the program will open.
    nvrtc_dir=""
    nvrtc_lib="$(ldd "$exe" | awk '$1 ~ /^libnvrtc\.so/ {print $3; exit}')"
    [ -n "$nvrtc_lib" ] && [ -f "$nvrtc_lib" ] && nvrtc_dir="$(dirname "$nvrtc_lib")"
    for extra in "$cudnn_root"/lib/libcudnn*.so.9* /usr/lib/x86_64-linux-gnu/libcudnn*.so.9* \
                 "${nvrtc_dir:-/nonexistent}"/libnvrtc-builtins*.so*; do
        [ -f "$extra" ] || continue
        [ -f "$out/lib/$(basename "$extra")" ] && continue
        cp -Lu "$extra" "$out/lib/" 2>/dev/null && say "bundled $(basename "$extra")"
    done
    # OpenBLAS is a hard link-time dependency and is not part of CUDA; the
    # Windows bundle carries it for the same reason.
    for blas in $(ldd "$exe" | awk '/libopenblas|libblas|liblapack/{print $3}'); do
        [ -f "$blas" ] && cp -Lu "$blas" "$out/lib/" && say "bundled $(basename "$blas")"
    done
    mkdir -p "$out/cuda/include"
    if [ "$cuda_inc" = /usr/include ] && command -v dpkg >/dev/null 2>&1; then
        # /usr/include is shared with the system, so take only what the toolkit
        # package owns rather than three hundred megabytes of everything.
        pkg="$(dpkg -S "$cuda_inc/cuda_runtime.h" 2>/dev/null | cut -d: -f1 | head -1)"
        dpkg -L "$pkg" 2>/dev/null | grep -E '^/usr/include/.*\.(h|hpp)$' |
            while read -r h; do cp -u "$h" "$out/cuda/include/" 2>/dev/null || true; done
    else
        cp -r "$cuda_inc/." "$out/cuda/include/"
    fi
    say "cuda headers: $(find "$out/cuda/include" -type f | wc -l) files"
    mkdir -p "$out/include/cccl"
    for tree in cuda thrust cub libcudacxx; do
        [ -d "$cccl/$tree" ] && cp -r "$cccl/$tree" "$out/include/cccl/"
    done
    say "headers: $(find "$out/include/cccl" -type f | wc -l) files from $cccl"
fi

cat > "$out/stemd" <<'LAUNCH'
#!/bin/bash
# Run stemd with its bundled CUDA, without installing anything.
here="$(cd "$(dirname "$0")" && pwd)"
[ -d "$here/lib" ] && export LD_LIBRARY_PATH="$here/lib:${LD_LIBRARY_PATH:-}"
# MLX builds kernels at run time and looks for the CUDA runtime headers under
# CUDA_HOME. Unset, every separation fails with "Can not find locations of CUDA
# headers" on a server that otherwise started perfectly and reported device:
# gpu. The CCCL headers beside these are found relative to the executable.
[ -d "$here/cuda/include" ] && export CUDA_HOME="$here/cuda"
# The driver is deliberately not bundled: it has to match the running kernel
# module, so shipping one machine's copy to another is a version mismatch
# rather than a GPU.
if [ -d "$here/lib" ] && ! /sbin/ldconfig -p 2>/dev/null | grep -q libcuda.so.1 &&
   [ ! -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
    echo "warning: libcuda.so.1 not found. That is the NVIDIA driver, and it is"
    echo "         not part of this bundle: it has to match the running kernel."
    echo "         With nouveau this will not start."
fi
exec "$here/bin/stemd-server" "$@"
LAUNCH
chmod +x "$out/stemd"

# Say what this is, in the directory itself. The most likely thing to happen to
# it is being found on a stick a year from now by someone who has to guess
# whether it is the thing to install.
cat > "$out/README" <<'DOC'
stemd, bundled for a machine you cannot install packages on
===========================================================

This directory carries its own CUDA runtime, its own cuDNN, and the headers MLX
needs to compile kernels while it runs. That is why it is well over a gigabyte.
It exists for one case: a live USB, or a machine where apt is not an option.

    ./stemd

On Debian or Ubuntu, use the .deb instead. It is about 55 MB, because it takes
those same libraries from the distribution, which means one copy of cuDNN on
the machine rather than two that can disagree about which engines to load. See
scripts/package-deb.sh.

Either way the NVIDIA driver comes from the machine, not from here: it has to
match the running kernel, so a copy of one machine's driver is a version
mismatch rather than a GPU. CUDA 13 needs 580 or later.
DOC

printf '\n'
say "packaged into $out"
du -sh "$out" | sed 's/^/  /'
say "this is the no-package-manager build; on Debian or Ubuntu prefer the .deb"
say "  scripts/package-deb.sh"
