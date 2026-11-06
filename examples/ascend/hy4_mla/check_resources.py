"""Source-level on-chip budget preflight; Python stdlib only, no NPU access.

Reads actual allocation shapes, aliases, version counts and fixed addresses.
The no-reuse sum is conservative for these explicitly laid-out buffers. This
does not run layout inference or prove synchronization/scheduling feasibility.
Compiler-inserted temporaries and final generated arenas still need inspection.
"""

import argparse
import ast
from collections import defaultdict
import json
import math
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SCOPES = {
    "alloc_shared": ("UB", 32, "kAscendUBMemoryBytes"),
    "alloc_l1": ("L1", 512, "kAscendL1MemoryBytes"),
    "alloc_l0a": ("L0A", 512, "kAscendL0ABMemoryBytes"),
    "alloc_l0b": ("L0B", 512, "kAscendL0ABMemoryBytes"),
    "alloc_l0c": ("L0C", 1024, "kAscendL0CMemoryBytes"),
    "alloc_ss": ("SS", 32, "kAscendSSMemoryBytes"),
}
DTYPE_BITS = {"uint8": 8, "bfloat16": 16, "float32": 32, "int32": 32, "uint32": 32, "uint64": 64}


def evaluate(node, env):
    # Expressions come from this trusted kernel source. No TileLang/module
    # objects are supplied, so dynamic IR expressions fail instead of guessing.
    return eval(compile(ast.Expression(node), "<kernel allocation>", "eval"), {"__builtins__": {}, "min": min}, env)  # noqa: S307


def inspect(topk=2048, pipeline=True, source=None, *, source_text=None, kv_format="bf16"):
    if topk <= 0 or topk % 64:
        raise ValueError("topk must be a positive multiple of 64")
    tree = ast.parse(source_text if source_text is not None else (source or HERE / "kernel.py").read_text(encoding="utf-8"))
    builder = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "hy4_sparse_attention")
    env = {
        "nq": 65,
        "nk": 257,
        "topk": topk,
        "scale": 1 / 16,
        "sink": True,
        "variable_lengths": True,
        "pipeline_stages": pipeline,
        "num_cores": 32,
        "q_strides": None,
        "kv_stride": None,
        "kv_format": kv_format,
        "index_stride": None,
        "out_strides": None,
    }
    for stmt in builder.body:
        if isinstance(stmt, ast.Assign):
            env[stmt.targets[0].id] = evaluate(stmt.value, env)
    body = next(n for n in builder.body if isinstance(n, ast.FunctionDef) and n.name == "attention_body")
    buffers, aliases, versions, addresses = {}, {}, {}, {}

    def visit(stmts):
        for stmt in stmts:
            if isinstance(stmt, ast.If):
                visit(stmt.body if evaluate(stmt.test, env) else stmt.orelse)
            elif isinstance(stmt, ast.With):
                # All allocations are in the launch block. SIMD blocks here
                # only initialize buffers and need no extra storage.
                if ast.unparse(stmt.items[0].context_expr.func) == "T.MixedKernel":
                    visit(stmt.body)
            elif isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                name = stmt.targets[0].id
                kind = ast.unparse(call.func).removeprefix("T.")
                kwargs = {k.arg: k.value for k in call.keywords}
                if kind in SCOPES:
                    shape = tuple(evaluate(call.args[0], env))
                    dtype = evaluate(call.args[1], env)
                    buffers[name] = {"shape": shape, "dtype": dtype, "scope": SCOPES[kind][0], "alignment": SCOPES[kind][1]}
                    env[name] = name
                elif kind == "view":
                    parent = aliases.get(call.args[0].id, call.args[0].id)
                    original = buffers[parent]
                    shape = tuple(evaluate(kwargs["shape"], env)) if "shape" in kwargs else original["shape"]
                    dtype = evaluate(kwargs["dtype"], env) if "dtype" in kwargs else original["dtype"]
                    if math.prod(shape) * DTYPE_BITS[dtype] != math.prod(original["shape"]) * DTYPE_BITS[original["dtype"]]:
                        raise ValueError(f"{name}: T.view must preserve bit count of {parent}")
                    aliases[name] = parent
                    env[name] = name
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                name = ast.unparse(call.func)
                if name == "T.annotate_buffer_versions":
                    versions.update(evaluate(call.args[0], env))
                elif name == "T.annotate_buffer_addresses":
                    addresses.update(evaluate(call.args[0], env))

    visit(body.body)
    # Fail closed if future edits add allocations outside the inspected launch
    # prefix (e.g. inside a macro or pipelined loop).
    all_allocs = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) in {f"T.{k}" for k in SCOPES}]
    # Conditional smaller/larger are counted syntactically but absent in the
    # direct-mask specialization.
    expected_count = len(buffers) + (2 if env["direct_query_masks"] else 0)
    if len(all_allocs) != expected_count:
        raise ValueError("Unaccounted allocation: extend source-level resource inspection")
    for name, parent in aliases.items():
        if name in versions and versions[name] != versions[parent]:
            raise ValueError(f"Conflicting alias versions: {name}, {parent}")
    text = (REPO / "src/ascend/target_utils.h").read_text(encoding="utf-8")
    constants = {name: int(a) * int(b) for name, a, b in re.findall(r"constexpr int64_t (\w+) = (\d+) \* (\d+);", text)}
    totals = defaultdict(int)
    intervals = defaultdict(list)
    rows = []
    for name, info in buffers.items():
        count = versions[name]
        size = math.prod(info["shape"]) * DTYPE_BITS[info["dtype"]] // 8
        alignment = info["alignment"]
        # Align every version independently, as required by DMA handoffs.
        stride = (size + alignment - 1) // alignment * alignment
        physical = stride * count
        totals[info["scope"]] += physical
        rows.append({"name": name, **info, "versions": count, "bytes": physical})
        if name in addresses:
            begin, end = addresses[name], addresses[name] + physical
            if begin % alignment:
                raise ValueError(f"Misaligned address for {name}")
            for other, lo, hi in intervals[info["scope"]]:
                if begin < hi and lo < end:
                    raise ValueError(f"Fixed address overlap: {name}, {other}")
            intervals[info["scope"]].append((name, begin, end))
    result = {"topk": topk, "pipeline": pipeline, "buffers": rows, "pools": {}}
    for scope, _, constant in SCOPES.values():
        peak = max((end for _, _, end in intervals[scope]), default=0)
        used = max(totals[scope], peak)
        capacity = constants[constant]
        result["pools"][scope] = {"no_reuse_bytes": totals[scope], "address_end": peak, "capacity": capacity, "fits": used <= capacity}
    result["fits"] = all(pool["fits"] for pool in result["pools"].values())
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--no-pipeline", action="store_true")
    parser.add_argument("--kv-format", choices=["bf16", "fp8_ds_mla"], default="bf16")
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    result = inspect(args.topk, not args.no_pipeline, args.source, kv_format=args.kv_format)
    if not args.details:
        result.pop("buffers")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["fits"] else 1)


if __name__ == "__main__":
    main()
