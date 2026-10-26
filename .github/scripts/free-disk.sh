#!/usr/bin/env bash
# Remove toolchains the hosted runner image carries and these jobs never use.
# The CUDA image and a staged package tree need about 25 GB.
set -euo pipefail
df -h / | tail -1
sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc /usr/local/.ghcup \
    /opt/hostedtoolcache/CodeQL /usr/local/share/boost /usr/share/swift
sudo docker image prune -af >/dev/null 2>&1 || true
df -h / | tail -1
