#!/usr/bin/env bash
# Build, or fetch and check, the prebuilt MLX for this machine's architecture.
#
#   prebuilt.sh archs          print MLX_CUDA_ARCHITECTURES for this architecture
#   prebuilt.sh build OUT      build MLX from source, pack it as OUT/<asset>
#   prebuilt.sh fetch DIR      download the pinned asset, check it, unpack into DIR
#
# Runs in the image built from packaging/linux/Dockerfile. The asset is
# mlx-cuda-linux-<arch>.tar.zst on the release MLX_PREBUILT_TAG.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck source=mlx-prebuilt.env
. "$root/packaging/linux/mlx-prebuilt.env"

arch="$(uname -m)"
archs_var="CUDA_ARCHS_$arch"
sha_var="SHA256_$arch"
archs="${!archs_var:?no CUDA_ARCHS_$arch in mlx-prebuilt.env}"
asset="mlx-cuda-linux-$arch.tar.zst"
repo="${GITHUB_REPOSITORY:-nsaintot/stemd}"

case "${1:-}" in
    archs)
        printf '%s\n' "$archs"
        ;;
    build)
        out="${2:?usage: prebuilt.sh build OUT}"
        mkdir -p "$out"
        tree="$(mktemp -d)"
        MLX_CUDA_ARCHITECTURES="$archs" MLX_PREBUILT_OUT="$tree" \
            cargo build --release --manifest-path "$root/Cargo.toml" -p mlx-sys
        tar -C "$tree" -c . | zstd -19 -T0 -q -o "$out/$asset" -f
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
        zstd -dc "$tmp" | tar -C "$dir" -x
        rm -f "$tmp"
        cat "$dir/mlx-prebuilt.txt"
        ;;
    *)
        echo "usage: prebuilt.sh archs | build OUT | fetch DIR" >&2
        exit 2
        ;;
esac
