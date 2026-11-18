#!/usr/bin/env bash
# Run the pytest selectors listed in the CSV's 测试用例 column.
# Relative CSV_PATH and LOG_DIR values are resolved from REPO_ROOT.
set -eu

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python}"
if [[ -z "${PYTHON:-}" ]] && ! command -v "$python_bin" >/dev/null 2>&1; then
    python_bin=python3
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
    printf 'ERROR: Python is not available: %s\n' "$python_bin" >&2
    exit 2
fi

exec "$python_bin" - "$script_dir" <<'PY'
import csv
import os
from collections import Counter
from importlib.util import find_spec
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory, mkdtemp
import xml.etree.ElementTree as ET


def read_selectors(csv_path):
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream, strict=True)
        if "测试用例" not in (rows.fieldnames or []):
            raise ValueError("CSV is missing the 测试用例 column")
        selectors = {}
        for row in rows:
            for line in (row["测试用例"] or "").splitlines():
                selector = line.strip()
                if selector and selector != "待覆盖":
                    selectors[selector] = None
        return selectors


def run_selector(selector, repo, xml_path, log, env):
    test_path = repo / selector.split("::", 1)[0]
    if not test_path.is_file():
        return "ERROR", f"test file not found: {test_path}"

    xml_path.unlink(missing_ok=True)
    command = [
        sys.executable, "-m", "pytest", "-vv", "-ra", "-s", "--color=no",
        "-o", "addopts=", "-o", "junit_family=xunit1", f"--junitxml={xml_path}",
        "--", selector,
    ]
    print(shlex.join(command), file=log, flush=True)
    result = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
    code = result.returncode

    if not xml_path.is_file():
        return "ERROR", f"pytest exit {code}, no JUnit report"

    issues = Counter()
    collected = 0
    for case in ET.parse(xml_path).iter("testcase"):
        if case.find("error") is not None:
            issues["error"] += 1
        # xunit1 includes the source line for real test cases. A module skipped
        # during import can produce a synthetic JUnit entry with no source line.
        if case.get("line") is None:
            continue
        collected += 1
        if case.find("failure") is not None:
            issues["failed"] += 1
        elif case.find("skipped") is not None:
            issues["skipped"] += 1
        elif case.find("error") is None:
            issues["passed"] += 1

    detail = ", ".join(f"{issues[key]} {key}" for key in ("passed", "failed", "skipped", "error"))
    if not collected:
        return "ERROR", f"no test cases collected/executed (pytest exit {code})"
    if issues["error"] or code not in (0, 1, 5):
        return "ERROR", f"{detail}, pytest exit {code}"
    if issues["failed"]:
        return "FAIL", detail
    # The repo's terminal summary can exit 5 for a run in which every
    # collected case was skipped. The JUnit case records distinguish this
    # from an unmatched selector, which has no real test case.
    if code in (0, 5) and issues["skipped"] == collected:
        return "SKIP", detail
    if code == 0 and issues["passed"]:
        return "PASS", detail
    return "ERROR", f"{detail}, pytest exit {code}"


def main():
    repo = Path(os.environ.get("REPO_ROOT") or sys.argv[1]).resolve()
    csv_path = repo / (os.environ.get("CSV_PATH") or "memory_management_layout_features.csv")
    log_dir = repo / (os.environ.get("LOG_DIR") or ".pytest_cache/memory_layout")
    if not (repo / "testing").is_dir() or not (repo / "pyproject.toml").is_file():
        raise ValueError(f"Invalid TileLang repo root: {repo}")
    if not csv_path.is_file():
        raise ValueError(f"CSV not found: {csv_path}")
    if find_spec("pytest") is None:
        raise ValueError(f"pytest is not available in {sys.executable}")

    selectors = read_selectors(csv_path)
    if not selectors:
        raise ValueError("CSV contains no runnable test selectors")

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(mkdtemp(prefix="run-", dir=log_dir)) / "pytest.log"
    counts = Counter()
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with log_path.open("w", encoding="utf-8") as log, TemporaryDirectory() as scratch:
        xml_path = Path(scratch) / "pytest.xml"

        def report(message):
            print(message, flush=True)
            print(message, file=log, flush=True)

        report(f"{len(selectors)} unique selectors | {repo}\nCSV: {csv_path}\nLog: {log_path}")
        for index, selector in enumerate(selectors, 1):
            print(f"\n[{index}/{len(selectors)}] {selector}", file=log, flush=True)
            try:
                status, detail = run_selector(selector, repo, xml_path, log, env)
            except (OSError, ET.ParseError) as error:
                status, detail = "ERROR", str(error)
            counts[status] += 1
            report(f"{status:5} {selector} ({detail})")

        summary = " | ".join(f"{key} {counts[key]}" for key in ("PASS", "FAIL", "SKIP", "ERROR"))
        report(f"\nSummary (selectors): {summary}\nLog: {log_path}")
    return 1 if counts["FAIL"] or counts["ERROR"] else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, csv.Error) as error:
        sys.exit(f"ERROR: {error}")
PY
