#!/usr/bin/env bash
# List the CUDA toolkit headers MLX's run-time kernel compiler reads, into
# packaging/linux/jit-headers.txt. stage.sh ships those and CCCL, and nothing
# else from the toolkit's include directory.
#
#   jit-headers.sh
#
# Runs in the image from packaging/linux/Dockerfile. Compiles every header MLX
# embeds for NVRTC, with the options MLX passes, for every compute capability
# CUDA 13 supports, and records the files NVRTC opens. Fails if one is a CCCL
# file outside the cuda/ and nv/ trees stage.sh ships. Run it again after an
# MLX or CUDA update.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
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
# MLX compiles for the device it finds, as sm_<cc> with an `a` suffix from
# compute capability 9.0 on, and passes no other option but include paths.
dev="$work/mlx/mlx/backend/cuda/device"
for h in "$dev"/*.cuh "$dev"/*.h; do
    printf '#include "mlx/backend/cuda/device/%s"\n#include <cooperative_groups.h>\n' \
        "$(basename "$h")" > "$work/tu.cu"
    for cc in 75 80 86 87 89 90a 100a 103a 110a 120a 121a; do
        LD_LIBRARY_PATH="$cuda/lib64" strace -f -e trace=openat -o "$work/trace.$cc.$(basename "$h")" \
            "$work/nvrtc" "$work/tu.cu" "--gpu-architecture=sm_$cc" --include-path="$work/mlx" \
            --include-path="$inc/cccl" --include-path="$inc" >/dev/null 2>&1
    done
done

opened="$(cat "$work"/trace.* | grep -v ENOENT | grep -oE "\"$inc/[^\"]+\"" | tr -d '"' \
    | sed "s#^$inc/##" | sort -u)"
outside="$(grep '^cccl/' <<<"$opened" | grep -vE '^cccl/(cuda|nv)/' || true)"
if [[ -n "$outside" ]]; then
    echo "ERROR: NVRTC opens CCCL files outside cuda/ and nv/; ship them in stage.sh:" >&2
    echo "$outside" >&2
    exit 1
fi
grep -v '^cccl/' <<<"$opened" > "$root/packaging/linux/jit-headers.txt"
wc -l < "$root/packaging/linux/jit-headers.txt"
