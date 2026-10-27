#!/usr/bin/env bash
# Separate a short generated mix on a machine without a GPU: the server starts,
# runs on the CPU and returns stems.
#
#   cpu-separation.sh SERVER CLI
set -euo pipefail
server="$1"
cli="$2"
work="$(mktemp -d)"
python3 "$(dirname "$0")/test-mix.py" "$work/mix.wav"

"$server" --headless --no-mdns --bind 127.0.0.1:7979 --cache-dir "$work/cache" >"$work/server.log" 2>&1 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true' EXIT
fail() {
    cat "$work/server.log"
    echo "ERROR: $1" >&2
    exit 1
}
for _ in $(seq 1 600); do
    grep -q listening "$work/server.log" && break
    kill -0 "$pid" 2>/dev/null || fail "the server exited"
    sleep 1
done
grep -q listening "$work/server.log" || fail "the server is not listening after 10 minutes"
grep -q "via mlx on cpu" "$work/server.log" || fail "the server is not on the CPU"
"$cli" --host 127.0.0.1:7979 --out "$work/stems" "$work/mix.wav" || fail "the separation failed"
ls -l "$work/stems"
