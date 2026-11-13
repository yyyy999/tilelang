#!/bin/bash
# Compile the single-AIV vexp kernel and run it under cannsim record.
set -euo pipefail
set +u
. /usr/local/Ascend/cann/set_env.sh
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-/root/vexp_kernel_bench}"
NITERS="${2:-64}"
ARCH="${ASCEND_NPU_ARCH:-dav-3510}"

mkdir -p "$OUT"
cp "$ROOT/vexp_kernel.asc" "$OUT/vexp_kernel.asc"
cd "$OUT"

INC="$ASCEND_HOME_PATH/include"
LIB="$ASCEND_HOME_PATH/lib64"
if [ ! -d "$INC" ]; then
  INC="$ASCEND_TOOLKIT_HOME/include"
  LIB="$ASCEND_TOOLKIT_HOME/lib64"
fi

echo "compiling npu-arch=$ARCH INC=$INC"
bisheng -O2 -std=c++17 --npu-arch="$ARCH" \
  -I"$INC" \
  vexp_kernel.asc -o vexp_kernel_bench \
  -L"$LIB" -Wl,-rpath,"$LIB" -lascendcl

echo "binary: $OUT/vexp_kernel_bench"
echo "running cannsim record n_iters=$NITERS"
rm -rf "$OUT/sim_out"
mkdir -p "$OUT/sim_out"
npusim record "$OUT/vexp_kernel_bench" -s Ascend950 -o "$OUT/sim_out" -g \
  -u "$NITERS"
echo "done. results under $OUT/sim_out"
echo
echo "=== vexp / vld / vst timing ==="
python3 "$ROOT/parse_trace.py" "$OUT/sim_out"
