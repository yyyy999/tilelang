#!/bin/bash
# Compile the Hy4 576-wide gather microbench and record one CANNsim run.
# Argument 1 is the Linux output directory. Argument 2 is 0 (16 pair copies)
# or 1 (one contiguous 32x576 copy).
set -euo pipefail
set +u
. /usr/local/Ascend/cann/set_env.sh
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-/root/hy4_gather_mte}"
MODE="${2:-0}"
ARCH="${ASCEND_NPU_ARCH:-dav-3510}"

mkdir -p "$OUT"
cp "$ROOT/gather_mte.asc" "$OUT/gather_mte.asc"
cd "$OUT"

INC="$ASCEND_HOME_PATH/include"
LIB="$ASCEND_HOME_PATH/lib64"
if [ ! -d "$INC" ]; then
  INC="$ASCEND_TOOLKIT_HOME/include"
  LIB="$ASCEND_TOOLKIT_HOME/lib64"
fi

echo "compiling npu-arch=$ARCH mode=$MODE"
bisheng -O2 -std=c++17 --npu-arch="$ARCH" \
  -I"$INC" \
  gather_mte.asc -o gather_mte_bench \
  -L"$LIB" -Wl,-rpath,"$LIB" -lascendcl

rm -rf "$OUT/sim_out"
mkdir -p "$OUT/sim_out"
npusim record "$OUT/gather_mte_bench" -s Ascend950 -o "$OUT/sim_out" -g -u "$MODE"
echo "done mode=$MODE under $OUT/sim_out"
