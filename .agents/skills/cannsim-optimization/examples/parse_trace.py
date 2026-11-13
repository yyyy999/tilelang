#!/usr/bin/env python3
"""Print RV_VEXP / VLD / VST timing from a cannsim Chrome-trace JSON."""
import argparse
import collections
import glob
import json
import os
import sys

INTERESTING = (
    "RV_VEXP",
    "RV_VDUP",
    "RV_VDUPS",
    "RV_VDUPV",
    "RV_VLD",
    "RV_VLDS",
    "RV_VLDI",
    "RV_VST",
    "RV_VSTS",
    "RV_VSTI",
    "VF",
    "MOV_SRC_TO_DST_ALIGNv2",
)
HEAD_TAIL = 8
FP32_LANES = 64
# CANN 9.2.0 Chrome traces store microseconds. One cycle is 1/1650 us:
# an fp32 vexp dur of 0.00969696... us is exactly 16 cycles.
CYCLE_PER_US = 1650.0


def load_events(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("traceEvents") or data.get("events") or []
    return [e for e in data if isinstance(e, dict) and e.get("ph") == "X"]


def issue_intervals(xs):
    return [round(xs[i]["ts"] - xs[i - 1]["ts"], 6) for i in range(1, len(xs))]


def _fractional(value):
    return isinstance(value, float) and abs(value - round(value)) > 1e-4


def trace_is_microseconds(events):
    for event in events:
        if _fractional(event.get("ts")) or _fractional(event.get("dur")):
            return True
    return False


def convert_microseconds_to_cycles(events):
    converted = []
    for event in events:
        event = dict(event)
        if "ts" in event:
            event["ts"] = round(float(event["ts"]) * CYCLE_PER_US, 6)
        if event.get("dur") is not None:
            event["dur"] = round(float(event["dur"]) * CYCLE_PER_US, 6)
        converted.append(event)
    return converted


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def median(xs):
    if not xs:
        return None
    ys = sorted(xs)
    mid = len(ys) // 2
    if len(ys) % 2:
        return float(ys[mid])
    return (ys[mid - 1] + ys[mid]) / 2.0


def print_event(e, prev_ts):
    ts = e.get("ts")
    dur = e.get("dur")
    extra = (e.get("args") or {}).get("extra_info", "").strip()
    ii = None if prev_ts is None else ts - prev_ts
    print(
        f"  ts={ts} dur={dur} end={ts + dur} ii={ii} "
        f"cat={e.get('cat')} tid={e.get('tid')} {extra}"
    )


def fit_ii(xs, warmup):
    """Steady-state II after dropping `warmup` leading issues (pipeline fill)."""
    if len(xs) < 2:
        return None
    iis = issue_intervals(xs)
    skip = min(max(warmup, 0), max(len(iis) - 1, 0))
    steady = iis[skip:]
    latency = xs[0].get("dur") or 0
    span = (xs[-1]["ts"] + xs[-1]["dur"]) - xs[0]["ts"]
    n = len(xs)
    # span ≈ latency + (n-1)*II  when the pipe stays full.
    ii_from_span = (span - latency) / (n - 1) if n > 1 else None
    return {
        "n": n,
        "latency": latency,
        "span": span,
        "mean_ii_all": mean(iis),
        "median_ii_all": median(iis),
        "warmup_issues": skip,
        "mean_ii_steady": mean(steady),
        "median_ii_steady": median(steady),
        "ii_hist_all": dict(sorted(collections.Counter(iis).items())),
        "ii_hist_steady": dict(sorted(collections.Counter(steady).items())),
        "ii_from_span": ii_from_span,
    }


def summarize(events, verbose_limit=16):
    names = collections.Counter(e.get("name") for e in events)
    print(
        "names: "
        + ", ".join(f"{name}={count}" for name, count in names.most_common())
    )
    print()
    by_name = {n: [] for n in INTERESTING}
    for e in events:
        name = e.get("name")
        if name in by_name:
            by_name[name].append(e)
    for name in INTERESTING:
        xs = sorted(by_name[name], key=lambda e: e.get("ts", 0))
        print(f"=== {name} n={len(xs)} ===")
        if not xs:
            print()
            continue
        dump = len(xs) <= verbose_limit
        if dump:
            prev = None
            for e in xs:
                print_event(e, prev)
                prev = e.get("ts")
        else:
            print(f"  (showing first/last {HEAD_TAIL})")
            for e in xs[:HEAD_TAIL]:
                print_event(e, None)
            print("  ...")
            for e in xs[-HEAD_TAIL:]:
                print_event(e, None)
        fit = fit_ii(xs, warmup=0)
        if fit:
            print(f"  span={fit['span']} latency={fit['latency']}")
            print(
                f"  mean_ii={fit['mean_ii_all']:.3f} "
                f"median_ii={fit['median_ii_all']:.3f} "
                f"ii_from_span={fit['ii_from_span']:.3f}"
            )
            print(f"  ii_hist={fit['ii_hist_all']}")
        print()
    if events:
        t0 = min(e["ts"] for e in events if "ts" in e)
        t1 = max(e["ts"] + e.get("dur", 0) for e in events if "ts" in e)
        print(f"kernel span ts {t0} .. {t1} = {t1 - t0}")

    for op_name in ("RV_VSTI", "RV_VSTS", "RV_VDUPS", "RV_VDUP", "RV_VDUPV",
                    "RV_VEXP"):
        xs = sorted(by_name.get(op_name, []), key=lambda e: e.get("ts", 0))
        if len(xs) < 2:
            if xs:
                print(f"{op_name} issues={len(xs)} (need >=2 to fit II)")
            continue
        latency = xs[0].get("dur") or 1
        warmup = min(int(latency), max(len(xs) // 4, 0))
        fit = fit_ii(xs, warmup=warmup)
        n = fit["n"]
        print()
        print(f"=== {op_name} throughput fit (fp32 lanes=64) ===")
        print(
            f"issues={n} latency={fit['latency']} occupancy_span={fit['span']} "
            f"warmup_dropped={fit['warmup_issues']}"
        )
        print(
            f"II all:    mean={fit['mean_ii_all']:.3f} "
            f"median={fit['median_ii_all']:.3f} hist={fit['ii_hist_all']}"
        )
        print(
            f"II steady: mean={fit['mean_ii_steady']:.3f} "
            f"median={fit['median_ii_steady']:.3f} hist={fit['ii_hist_steady']}"
        )
        print(f"II from span-latency: {fit['ii_from_span']:.3f}")
        for label, ii in (
            ("mean_all", fit["mean_ii_all"]),
            ("median_steady", fit["median_ii_steady"]),
            ("span_fit", fit["ii_from_span"]),
        ):
            if ii and ii > 0:
                print(f"  {label}: {FP32_LANES / ii:.3f} ops/cycle  (64/II)")
        print(
            f"  occupancy: {n * FP32_LANES / fit['span']:.3f} ops/cycle  (n*64/span)"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trace",
        nargs="?",
        help="trace_core0.json, or a cannsim output dir to search",
    )
    args = parser.parse_args()
    path = args.trace
    if path is None:
        cands = sorted(
            glob.glob("/root/vexp_bench/sim_out/cannsim_*/report/trace_core0.json")
        )
        if not cands:
            print("no default trace under /root/vexp_bench/sim_out", file=sys.stderr)
            return 1
        path = cands[-1]
    elif os.path.isdir(path):
        cands = sorted(glob.glob(os.path.join(path, "**/trace_core0.json"), recursive=True))
        if not cands:
            print(f"no trace_core0.json under {path}", file=sys.stderr)
            return 1
        path = cands[-1]
    print(f"trace: {path}")
    events = load_events(path)
    if trace_is_microseconds(events):
        print(
            "units: cycles, converted from CANN 9.2.0 chrome microseconds "
            f"by * {CYCLE_PER_US:.0f}"
        )
        events = convert_microseconds_to_cycles(events)
    else:
        print("units: cycles, trace values are already integral")
    summarize(events)
    return 0


if __name__ == "__main__":
    sys.exit(main())
