#!/usr/bin/env bash
set -euo pipefail
sharing_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_dir="${TILELANG_SHARING_VENV:-/root/tilelang-venv}"
compiler_dir="${TILELANG_SHARING_COMPILER:-/root/tilelang}"
source "$venv_dir/bin/activate"
cd "$compiler_dir"
for case_name in minimal_baseline minimal_offset fa; do
    output_dir="$sharing_dir/artifacts/$case_name"
    mkdir -p "$output_dir"
    case_args=(--case minimal)
    if [[ "$case_name" == minimal_offset ]]; then
        case_args+=(--offset)
    elif [[ "$case_name" == fa ]]; then
        case_args=(--case fa)
    fi
    python "$sharing_dir/dump_ir.py" "${case_args[@]}" --out "$output_dir" 2>&1 | tee "$output_dir/run.log"
done

