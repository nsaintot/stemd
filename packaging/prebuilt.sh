#!/usr/bin/env bash
# Build, or fetch and check, the prebuilt MLX for this platform.
#
#   prebuilt.sh archs          print MLX_CUDA_ARCHITECTURES for this platform
#   prebuilt.sh build OUT      build MLX from source, pack it as OUT/<asset>
#   prebuilt.sh fetch DIR      download the pinned asset, check it, unpack into DIR
#
# On Linux it runs in the image built from packaging/linux/Dockerfile; on
# Windows in Git Bash, after .github/scripts/windows-cuda.ps1. The asset is
# mlx-cuda-<os>-<arch>.tar.zst on Linux and .zip on Windows, attached to the
# release MLX_PREBUILT_TAG.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=SCRIPTDIR/mlx-prebuilt.env
. "$root/packaging/mlx-prebuilt.env"

case "$(uname -s)" in
    Linux) os=linux; arch="$(uname -m)"; ext=tar.zst ;;
    MINGW*|MSYS*) os=windows; arch=x86_64; ext=zip ;;
    *) echo "ERROR: no prebuilt MLX for $(uname -s)" >&2; exit 1 ;;
esac
# Paths handed to cargo and 7-Zip, which are Windows programs on Windows.
native() { if [[ "$os" == windows ]]; then cygpath -w "$1"; else printf '%s' "$1"; fi; }

platform="${os}_$arch"
archs_var="CUDA_ARCHS_$platform"
sha_var="SHA256_$platform"
archs="${!archs_var:?no CUDA_ARCHS_$platform in mlx-prebuilt.env}"
asset="mlx-cuda-$os-$arch.$ext"
repo="${GITHUB_REPOSITORY:-nsaintot/stemd}"

case "${1:-}" in
    archs)
        printf '%s\n' "$archs"
        ;;
    build)
        out="${2:?usage: prebuilt.sh build OUT}"
        mkdir -p "$out"
        tree="$(mktemp -d)"
        MLX_CUDA_ARCHITECTURES="$archs" MLX_PREBUILT_OUT="$(native "$tree")" \
            cargo build --release --manifest-path "$(native "$root/Cargo.toml")" -p mlx-sys
        rm -f "$out/$asset"
        if [[ "$os" == windows ]]; then
            (cd "$tree" && 7z a -tzip -mx=9 "$(native "$out/$asset")" . >/dev/null)
        else
            tar -C "$tree" -c . | zstd -19 -T0 -q -o "$out/$asset"
        fi
        rm -rf "$tree"
        sha256sum "$out/$asset"
        ;;
    fetch)
        dir="${2:?usage: prebuilt.sh fetch DIR}"
        want="${!sha_var}"
        [[ -n "$want" ]] || { echo "ERROR: no $sha_var in mlx-prebuilt.env" >&2; exit 1; }
        mkdir -p "$dir"
        tmp="$(mktemp)"
        curl -fsSL -o "$tmp" "https://github.com/$repo/releases/download/$MLX_PREBUILT_TAG/$asset"
        got="$(sha256sum "$tmp" | cut -d' ' -f1)"
        [[ "$got" == "$want" ]] || { echo "ERROR: $asset is $got, expected $want" >&2; exit 1; }
        if [[ "$os" == windows ]]; then
            7z x -y -o"$(native "$dir")" "$(native "$tmp")" >/dev/null
        else
            zstd -dc "$tmp" | tar -C "$dir" -x
        fi
        rm -f "$tmp"
        cat "$dir/mlx-prebuilt.txt"
        ;;
    *)
        echo "usage: prebuilt.sh archs | build OUT | fetch DIR" >&2
        exit 2
        ;;
esac
