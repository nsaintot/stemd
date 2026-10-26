# The Linux build and packaging image, for amd64 and arm64.
#
# Ubuntu 22.04 sets the glibc floor at 2.35: the packages install on Ubuntu
# 22.04 and later, Debian 12 and later, and Fedora 36 and later. The CUDA and
# cuDNN versions here are the ones the packages bundle.
#
#   docker build --platform linux/arm64 -t stemd-linux packaging/linux
FROM nvidia/cuda:13.3.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config ca-certificates curl file git gnupg \
        python3-pip patchelf zstd gettext-base desktop-file-utils ccache \
        liblapacke-dev libopenblas-dev liblapack-dev \
        libavahi-compat-libdnssd-dev \
    && rm -rf /var/lib/apt/lists/*

# bindgen needs libclang 16 or later: clang 14 rejects the _Float16 and __bf16
# in mlx-c's half.h. Jammy stops at 15.
RUN curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
        | gpg --dearmor -o /usr/share/keyrings/llvm.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" \
        > /etc/apt/sources.list.d/llvm.list \
    && apt-get update && apt-get install -y --no-install-recommends libclang-18-dev clang-18 \
    && rm -rf /var/lib/apt/lists/*
ENV LIBCLANG_PATH=/usr/lib/llvm-18/lib

# CMake 3.25 fails find_package on CUDA::nvToolsExt, which CUDA 12 removed.
RUN pip3 install --no-cache-dir cmake==3.31.6

RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

# nfpm writes the deb and the rpm from one recipe; appimagetool writes the
# AppImage. appimagetool is itself an AppImage and a container has no FUSE, so
# it is unpacked once here.
ARG NFPM_VERSION=2.41.1
ARG APPIMAGETOOL_VERSION=1.9.0
RUN set -eux; \
    case "$(dpkg --print-architecture)" in \
        amd64) nfpm_arch=x86_64; ai_arch=x86_64 ;; \
        arm64) nfpm_arch=arm64;  ai_arch=aarch64 ;; \
        *) echo "unsupported architecture" >&2; exit 1 ;; \
    esac; \
    curl -fsSL "https://github.com/goreleaser/nfpm/releases/download/v${NFPM_VERSION}/nfpm_${NFPM_VERSION}_Linux_${nfpm_arch}.tar.gz" \
        | tar -xz -C /usr/local/bin nfpm; \
    curl -fsSL -o /tmp/appimagetool \
        "https://github.com/AppImage/appimagetool/releases/download/${APPIMAGETOOL_VERSION}/appimagetool-${ai_arch}.AppImage"; \
    chmod +x /tmp/appimagetool; \
    cd /opt && /tmp/appimagetool --appimage-extract >/dev/null && mv squashfs-root appimagetool.d; \
    rm /tmp/appimagetool; \
    printf '#!/bin/sh\nexec /opt/appimagetool.d/AppRun "$@"\n' > /usr/local/bin/appimagetool; \
    chmod +x /usr/local/bin/appimagetool

# MLX's FindCUDNN wants include/ and lib/ under one root, and the cuDNN packages
# put both in the multiarch directories.
RUN multiarch="$(gcc -print-multiarch)" \
    && mkdir -p /opt/cudnn \
    && ln -s "/usr/include/$multiarch" /opt/cudnn/include \
    && ln -s "/usr/lib/$multiarch" /opt/cudnn/lib

# The build links -lcuda against the toolkit's stub; the driver comes from the
# machine that runs the result.
ENV CUDA_PATH=/usr/local/cuda \
    LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
    MLX_BUILD_CUDA=1 \
    CUDNN_PATH=/opt/cudnn

WORKDIR /work
ENTRYPOINT ["/bin/bash", "-lc"]
