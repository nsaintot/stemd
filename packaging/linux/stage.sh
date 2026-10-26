#!/usr/bin/env bash
# Assemble the tree that every Linux package ships.
#
#   stage.sh [--out DIR] [--bin DIR]
#
# The deb and the rpm install it at /opt/stemd; the AppImage mounts it.
#
#   opt/stemd/stemd           launcher: sets CUDA_HOME, execs bin/stemd-server
#   opt/stemd/bin/            stemd-server, stemd-cli
#   opt/stemd/lib/            CUDA, cuDNN, BLAS and avahi-compat, vendored
#   opt/stemd/include/cccl/   CCCL headers for MLX's run-time kernel compiler
#   opt/stemd/cuda/include/   CUDA runtime headers for the same
#
# Runs in the image built from packaging/linux/Dockerfile, after
# `cargo build --release -p stemd-server -p stemd-cli`.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"

out="$root/dist/linux"
bin="${CARGO_TARGET_DIR:-$root/target}/release"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) out="$2"; shift 2 ;;
        --bin) bin="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

say() { printf '  %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

cuda="${CUDA_PATH:-/usr/local/cuda}"
cudnn="${CUDNN_PATH:-/opt/cudnn}"
case "$(uname -m)" in
    x86_64) cuda_target=x86_64-linux ;;
    aarch64) cuda_target=sbsa-linux ;;
    *) die "unsupported architecture $(uname -m)" ;;
esac

exe="$bin/stemd-server"
[[ -f "$exe" ]] || die "no $exe. Run cargo build --release -p stemd-server -p stemd-cli first."
[[ -f "$bin/stemd-cli" ]] || die "no $bin/stemd-cli."

# A CPU-only MLX links, starts and runs about 190x slower, so the binary is
# checked rather than the environment that built it.
needed="$(readelf -d "$exe" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p')"
for lib in libcudart.so.13 libcublasLt.so.13 libnvrtc.so.13 libcudnn.so.9; do
    grep -qx "$lib" <<<"$needed" || die "$exe does not link $lib: MLX was built without CUDA."
done

prefix="$out/opt/stemd"
rm -rf "$out"
mkdir -p "$prefix/bin" "$prefix/lib" "$prefix/include/cccl" "$prefix/cuda/include" \
         "$prefix/share/doc/stemd"
install -m 0755 "$exe" "$bin/stemd-cli" "$prefix/bin/"

# --- libraries -----------------------------------------------------------
#
# The driver, the display stack and the C++ runtime come from the machine;
# everything else is vendored. The host-owned list matches nfpm.yaml.in.
host_owned() {
    case "$1" in
        libcuda.so*|libnvidia-*.so*) return 0 ;;
        libGL.so*|libGLX.so*|libGLdispatch.so*|libEGL.so*|libOpenGL.so*|libdrm.so*|\
        libX11*.so*|libxcb*.so*|libXau.so*|libXdmcp.so*|libX[a-z]*.so*|\
        libwayland-*.so*|libxkbcommon*.so*) return 0 ;;
        libdbus-1.so*|libsystemd.so*) return 0 ;;
        ld-linux*.so*|libc.so*|libm.so*|libdl.so*|libpthread.so*|librt.so*|\
        libresolv.so*|libutil.so*|libgcc_s.so*|libstdc++.so*|libatomic.so*) return 0 ;;
    esac
    return 1
}

# ldd has to find a libcuda.so.1 to walk past it; the toolkit's stub serves.
stubs="$(mktemp -d)"
trap 'rm -rf "$stubs"' EXIT
ln -s "$cuda/lib64/stubs/libcuda.so" "$stubs/libcuda.so.1"
export LD_LIBRARY_PATH="$stubs:$cuda/lib64:$cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

soname() { readelf -d "$1" | sed -n 's/.*(SONAME).*\[\(.*\)\]/\1/p'; }

# cuDNN opens its engine libraries itself and NVRTC opens its builtins, so no
# NEEDED entry names them. They go into the walk so their own dependencies come
# too. Copied under their soname, which is the name they are opened by.
queue=("$exe")
for f in "$cudnn"/lib/libcudnn*.so.9 "$cuda"/lib64/libnvrtc-builtins.so; do
    [[ -f "$f" ]] || continue
    name="$(soname "$f")"
    [[ -f "$prefix/lib/$name" ]] && continue
    cp -L "$f" "$prefix/lib/$name"
    queue+=("$prefix/lib/$name")
done

# Direct dependencies only. ldd prints the whole closure, which would also
# vendor the dependencies of host-owned libraries such as libsystemd, and those
# must stay the host's.
missing=()
seen=" "
while [[ ${#queue[@]} -gt 0 ]]; do
    target="${queue[0]}"
    queue=("${queue[@]:1}")
    resolved="$(ldd "$target" | awk '$2 == "=>" {print $1, ($3 == "not" ? "" : $3)}')"
    for name in $(readelf -d "$target" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p'); do
        host_owned "$name" && continue
        [[ "$seen" == *" $name "* ]] && continue
        seen="$seen$name "
        path="$(awk -v n="$name" '$1 == n {print $2}' <<<"$resolved")"
        if [[ -z "$path" || ! -f "$path" ]]; then
            missing+=("$name")
            continue
        fi
        [[ -f "$prefix/lib/$name" ]] || cp -L "$path" "$prefix/lib/$name"
        queue+=("$prefix/lib/$name")
    done
done
[[ ${#missing[@]} -eq 0 ]] || die "not found, so not vendored: $(printf '%s ' "${missing[@]}" | tr ' ' '\n' | sort -u | xargs)"

# The executables carry DT_RPATH rather than DT_RUNPATH: it also covers the
# libraries that cuDNN and BLAS load. NVIDIA's libraries already look beside
# themselves ($ORIGIN) and the EULA forbids modifying them; the rest get the
# same RUNPATH here.
for f in "$prefix"/bin/*; do
    patchelf --force-rpath --set-rpath '$ORIGIN/../lib' "$f"
done
for f in "$prefix"/lib/*; do
    case "$(basename "$f")" in
        libcu*|libnv*) ;;
        *) chmod u+w "$f"; patchelf --set-rpath '$ORIGIN' "$f" ;;
    esac
done
chmod 0644 "$prefix"/lib/*
say "vendored $(ls "$prefix/lib" | wc -l) libraries, $(du -sh "$prefix/lib" | cut -f1)"

# --- headers -------------------------------------------------------------
#
# MLX compiles kernels while it runs. It finds CCCL at <bin>/../include/cccl
# and the runtime headers under CUDA_HOME, which the launcher sets. Only the
# toolkit headers in jit-headers.txt are shipped; jit-headers.sh lists them.
inc="$cuda/targets/$cuda_target/include"
for tree in cuda nv; do
    cp -r "$inc/cccl/$tree" "$prefix/include/cccl/"
done
while read -r h; do
    [[ -n "$h" ]] || continue
    [[ -f "$inc/$h" ]] || die "$inc/$h, listed in jit-headers.txt, does not exist"
    install -D -m 0644 "$inc/$h" "$prefix/cuda/include/$h"
done < "$root/packaging/linux/jit-headers.txt"
say "headers: $(find "$prefix/include" "$prefix/cuda" -type f | wc -l) files"

# --- launcher ------------------------------------------------------------
#
# MLX derives its CCCL path from argv[0] without resolving it, so the server is
# exec'd by absolute path, never through a symlink or by name.
cat > "$prefix/stemd" <<'EOF'
#!/bin/sh
here="$(dirname "$(readlink -f "$0")")"
export CUDA_HOME="$here/cuda"
if ! /sbin/ldconfig -p 2>/dev/null | grep -q 'libcuda\.so\.1 '; then
    echo "stemd: libcuda.so.1 not found. Install the NVIDIA driver, 580 or later." >&2
fi
exec "$here/bin/stemd-server" "$@"
EOF
chmod 0755 "$prefix/stemd"

# --- desktop integration and notices -------------------------------------
mkdir -p "$out/usr/share/applications" "$out/usr/share/icons/hicolor/256x256/apps"
install -m 0644 "$root/packaging/linux/stemd.desktop" "$out/usr/share/applications/"
install -m 0644 "$root/resources/stemd-icon-win-256.png" \
    "$out/usr/share/icons/hicolor/256x256/apps/stemd.png"
install -m 0644 "$root/LICENSE-MIT" "$root/LICENSE-APACHE" \
    "$root/packaging/linux/THIRD-PARTY" "$prefix/share/doc/stemd/"

# --- check ---------------------------------------------------------------
unresolved="$(LD_LIBRARY_PATH="$stubs" ldd "$prefix/bin/stemd-server" | grep 'not found' || true)"
[[ -z "$unresolved" ]] || die "unresolved after staging: $unresolved"
LD_LIBRARY_PATH="$stubs" "$prefix/bin/stemd-server" --version >/dev/null ||
    die "the staged stemd-server does not start"
say "staged $(du -sh "$out" | cut -f1) at $out"
