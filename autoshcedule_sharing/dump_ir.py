"""Capture sharing IR with the compiler imported by the active environment.

Activate tilelang-venv and run from its matching /root/tilelang checkout.
The input files and artifacts remain in the sharing folder.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
PASSES = [
    "tl.MaterializeScheduleUnits", "tl.AnnotateMultiBufferEligible",
    "tl.EstimateLatency", "tl.AutoSchedule", "tl.AssignCore",
    "tl.PrepareMultiBuffer", "tl.ResolveCore", "tl.InsertSync",
    "tl.MaterializeMultiBuffer", "tl.LowerScheduledTIR",
]


def git_revision(path):
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main(default_case="minimal"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("minimal", "fa"), default=default_case)
    parser.add_argument("--offset", action="store_true")
    parser.add_argument("--version-mode", choices=("counter", "iteration"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--trace", action="store_true", help="Also save full HTML trace.")
    args = parser.parse_args()
    if args.case == "fa" and (args.offset or args.version_mode):
        parser.error("--offset and --version-mode apply only to minimal.")
    version_mode = args.version_mode or ("iteration" if args.offset else "counter")
    name = "fa" if args.case == "fa" else "minimal_offset" if args.offset else "minimal_baseline"
    out = (args.out or HERE / "artifacts" / name).resolve()
    out.mkdir(parents=True, exist_ok=True)

    import tilelang
    from tilelang import tvm
    from tvm import tirx
    import z3

    compiler_root = Path(tilelang.__file__).resolve().parent.parent
    manifest = {
        "case": name,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "python": sys.executable,
        "working_directory": str(Path.cwd()),
        "tilelang_python": tilelang.__file__,
        "tvm_python": tvm.__file__,
        "compiler_checkout": str(compiler_root),
        "compiler_revision": git_revision(compiler_root),
        "sharing_checkout": str(HERE.parent),
        "sharing_revision": git_revision(HERE.parent),
        "z3_version": z3.get_version_string(),
        "enable_device_compile": False,
        "input_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [HERE / "minimal.py", HERE / "fa.py", HERE / "fa_core.py"]
        },
        "parameters": (
            {"q_len": 256, "kv_len": 512, "head_dim": 128, "num_blocks": 1, "manual_schedule": False}
            if args.case == "fa"
            else {"iterations": 4, "tile": 64, "fixed_l1_versions": 3,
                  "enable_offset": args.offset, "version_mode": version_mode}
        ),
        "snapshots": [],
    }

    def save_manifest():
        (out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )

    save_manifest()

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name not in PASSES:
                return
            number = PASSES.index(info.name) + 1
            filename = f"{number:02d}_{info.name.removeprefix('tl.')}.tir"
            script = mod.script()
            (out / filename).write_text(script + "\n", encoding="utf-8")
            stages = []
            tasks = []

            def visit(node):
                if not isinstance(node, tirx.AttrStmt):
                    return
                if node.attr_key == "tl.schedule_unit":
                    stages.append(int(node.node["stage"]))
                elif node.attr_key == "tl.ascend_task":
                    tasks.append({str(k): str(v) for k, v in node.node.items()})

            for func in mod.functions.values():
                if isinstance(func, tirx.PrimFunc):
                    tirx.stmt_functor.post_order_visit(func.body, visit)
            manifest["snapshots"].append({
                "pass": info.name, "file": filename, "lines": len(script.splitlines()),
                "schedule_unit_stages": stages, "task_metadata": tasks,
            })
            save_manifest()
            print(f"[IR] {info.name} -> {filename}", flush=True)

    try:
        if args.case == "minimal":
            from minimal import make_program
            program = make_program(enable_offset=args.offset, version_mode=version_mode)
        else:
            from fa import make_program
            program = make_program()
        (out / "00_frontend.tir").write_text(program.script() + "\n", encoding="utf-8")
        if args.trace:
            from tilelang.tools import lower_trace
            lower_trace.enable(mode="html", trace_dir=str(out / "trace"), codegen_output=None)
        with tvm.transform.PassContext(instruments=[Capture()]):
            artifact = tilelang.lower(program, target="ascend", enable_device_compile=False)
        (out / "kernel.cpp").write_text(artifact.kernel_source, encoding="utf-8")
        seen = {item["pass"] for item in manifest["snapshots"]}
        missing = set(PASSES) - seen
        if missing:
            raise RuntimeError(f"Expected passes not observed: {sorted(missing)}")
        manifest["status"] = "success"
        manifest["source_lines"] = len(artifact.kernel_source.splitlines())
        print(f"[OK] {name}: {len(seen)} stage snapshots + frontend + kernel source", flush=True)
        print(f"[OUT] {out}", flush=True)
    except Exception:
        manifest["status"] = "failed"
        manifest["error"] = traceback.format_exc()
        (out / "error.txt").write_text(manifest["error"], encoding="utf-8")
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_manifest()


if __name__ == "__main__":
    main()

