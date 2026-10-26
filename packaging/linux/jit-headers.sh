#!/usr/bin/env bash
# List the CUDA toolkit headers MLX's run-time kernel compiler reads, into
# packaging/linux/jit-headers.txt. stage.sh ships those and CCCL, and nothing
# else from the toolkit's include directory.
#
#   jit-headers.sh
#
# Runs in the image from packaging/linux/Dockerfile. Compiles every header MLX
# embeds for NVRTC, for every GPU architecture in mlx-prebuilt.env, and records
# the files NVRTC opens. Run it again after an MLX or CUDA update.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck source=mlx-prebuilt.env
. "$root/packaging/linux/mlx-prebuilt.env"
mlx_tag="$(sed -n 's/^const MLX_TAG: &str = "\(.*\)";/\1/p' "$root/vendor/mlx-rs-stemd/mlx-sys/build.rs")"
cuda="${CUDA_PATH:-/usr/local/cuda}"
case "$(uname -m)" in
    x86_64) inc="$cuda/targets/x86_64-linux/include" ;;
    aarch64) inc="$cuda/targets/sbsa-linux/include" ;;
esac
command -v strace >/dev/null || { apt-get update -qq && apt-get install -y -qq strace >/dev/null; }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
git clone -q --depth 1 --branch "$mlx_tag" https://github.com/ml-explore/mlx.git "$work/mlx"

# Compile argv[1] with NVRTC and the remaining arguments as options.
cat > "$work/nvrtc.c" <<'EOF'
#include <nvrtc.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb");
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    rewind(f);
    char *src = malloc(n + 1);
    src[fread(src, 1, n, f)] = 0;
    nvrtcProgram p;
    nvrtcCreateProgram(&p, src, "headers.cu", 0, NULL, NULL);
    nvrtcCompileProgram(p, argc - 2, (const char **)argv + 2);
    return 0;
}
EOF
gcc "$work/nvrtc.c" -I"$cuda/include" -L"$cuda/lib64" -lnvrtc -o "$work/nvrtc"

# One translation unit per header, as MLX's kernels include them. A header
# compiled on its own can fail; only the files opened matter.
dev="$work/mlx/mlx/backend/cuda/device"
# shellcheck disable=SC2154
archs="$CUDA_ARCHS_x86_64;$CUDA_ARCHS_aarch64"
for h in "$dev"/*.cuh "$dev"/*.h; do
    printf '#include "mlx/backend/cuda/device/%s"\n#include <cooperative_groups.h>\n' \
        "$(basename "$h")" > "$work/tu.cu"
    for a in ${archs//;/ }; do
        LD_LIBRARY_PATH="$cuda/lib64" strace -f -e trace=openat -o "$work/trace.$a.$(basename "$h")" \
            "$work/nvrtc" "$work/tu.cu" --include-path="$work/mlx" --include-path="$inc/cccl" \
            --include-path="$inc" -std=c++20 -default-device "-arch=sm_$a" >/dev/null 2>&1
    done
done

cat "$work"/trace.* | grep -v ENOENT | grep -oE "\"$inc/[^\"]+\"" | tr -d '"' \
    | sed "s#^$inc/##" | grep -v '^cccl/' | sort -u > "$root/packaging/linux/jit-headers.txt"
wc -l < "$root/packaging/linux/jit-headers.txt"
