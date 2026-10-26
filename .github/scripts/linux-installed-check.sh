#!/usr/bin/env bash
# Run where the deb or the rpm was just installed: every library resolves and
# the server starts. /dist/stub/libcuda.so.1 stands in for the driver.
set -euo pipefail
export LD_LIBRARY_PATH=/dist/stub
if ldd /opt/stemd/bin/stemd-server | grep "not found"; then
    echo "ERROR: stemd-server has unresolved libraries" >&2
    exit 1
fi
stemd-server --version
stemd-cli --version
echo "installed package passed"
