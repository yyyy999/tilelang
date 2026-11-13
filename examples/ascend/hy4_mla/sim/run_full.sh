#!/bin/bash
# Simulate the compiled Hy4 kernel with npusim.
# The Python program is examples/ascend/hy4_mla/sim/run_full.py.
# Argument 1 is the Linux output directory.
set -euo pipefail
set +u
. /usr/local/Ascend/cann/set_env.sh
set -u

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
if [ ! -d "$ROOT/build/lib" ] && [ -d /root/tilelang-hy4/build/lib ]; then
  ROOT=/root/tilelang-hy4
fi
OUT="${1:-/root/hy4_full_sim}"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
rm -rf "$OUT"
mkdir -p "$OUT"
npusim record \
  "/root/cannsim-venv/bin/python $ROOT/examples/ascend/hy4_mla/sim/run_full.py" \
  -s Ascend950 \
  -o "$OUT" \
  -g
echo "done. results under $OUT"
