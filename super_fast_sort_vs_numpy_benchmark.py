#!/usr/bin/env python3


import numpy as np
import time
import sys
import platform
import random
import argparse
from pathlib import Path


# ============================================================
# CYTHON IMPORT — Zero-overhead wrapper
# ============================================================
try:
    from super_fast_sort import super_fast_sort
except ImportError as e:
    print("❌ Failed to import super_fast_sort Cython extension")
    print(e)
    sys.exit(1)


# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "MAX_N": 10_000_000,
}


BAR_WIDTH = 40
HIST_BUCKETS = 40
HIST_HEIGHT = 10  # rows tall for the histogram


# ============================================================
# ARGUMENTS
# ============================================================
parser = argparse.ArgumentParser()


parser.add_argument("--time", type=int, help="Stop after N seconds")
parser.add_argument("--iter", type=int, help="Stop after N iterations")
parser.add_argument(
    "--no-fail",
    action="store_true",
    help="Continue even if a failure occurs",
)
parser.add_argument(
    "--detailed-mode",
    action="store_true",
)
parser.add_argument(
    "--max-n",
    type=int,
    help="Override the maximum array size",
)
parser.add_argument(
    "--distr",
    type=str,
    help="Test only the named distribution (e.g. uniform, sorted, zipf, …)",
)
parser.add_argument(
    "--size-hist",
    action="store_true",
    help="Show the win-rate histogram by array size",
)


ARGS = parser.parse_args()


if ARGS.max_n and ARGS.max_n > 0:
    CONFIG["MAX_N"] = ARGS.max_n


# ============================================================
# DATA GENERATORS
# ============================================================


def uniform(n):
    return np.random.randint(0, 2**32, n, dtype=np.uint32)


def sorted_asc(n):
    return np.arange(n, dtype=np.uint32)


def sorted_desc(n):
    return np.arange(n, dtype=np.uint32)[::-1].copy()


def sawtooth(n):
    if n == 0:
        return np.empty(0, np.uint32)
    chunk = max(1, n // 8)
    arr = np.arange(n, dtype=np.uint32)
    for i in range(0, n, chunk):
        if (i // chunk) & 1:
            arr[i:i+chunk] = arr[i:i+chunk][::-1]
    return arr


def pipe_organ(n):
    half = n // 2
    return np.concatenate((
        np.arange(0, half * 2, 2, dtype=np.uint32),
        np.arange(1, (n - half) * 2, 2, dtype=np.uint32)[::-1],
    ))


def zipfian(n):
    return np.random.zipf(2.0, n).astype(np.uint32)


def sparse_outliers(n):
    arr = np.zeros(n, dtype=np.uint32)
    if n:
        k = max(1, n // 100)
        idx = np.random.choice(n, k, replace=False)
        arr[idx] = np.random.randint(2**31, 2**32, size=len(idx), dtype=np.uint32)
    return arr


def constant(n):
    return np.full(n, np.random.randint(0, 2**32, dtype=np.uint32), dtype=np.uint32)


def two_values(n):
    return np.random.choice(
        [np.random.randint(0, 2**32), np.random.randint(0, 2**32)],
        size=n,
    ).astype(np.uint32)


DISTS = [
    ("uniform",    uniform),
    ("sorted",     sorted_asc),
    ("reverse",    sorted_desc),
    ("sawtooth",   sawtooth),
    ("pipe",       pipe_organ),
    ("zipf",       zipfian),
    ("sparse",     sparse_outliers),
    ("constant",   constant),
    ("two_values", two_values),
]

DIST_NAMES = [name for name, _ in DISTS]


# ============================================================
# VALIDATE --distr
# ============================================================
if ARGS.distr and ARGS.distr not in DIST_NAMES:
    print(f"❌ Unknown distribution: '{ARGS.distr}'")
    print(f"   Available: {', '.join(DIST_NAMES)}")
    sys.exit(1)

ACTIVE_DISTS = [(name, fn) for name, fn in DISTS if not ARGS.distr or name == ARGS.distr]


# ============================================================
# UTILS
# ============================================================


def format_runtime(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def make_bar(win, normal, fail, total):
    if total == 0:
        return "░" * BAR_WIDTH

    w = round((win    / total) * BAR_WIDTH)
    n = round((normal / total) * BAR_WIDTH)
    f = BAR_WIDTH - w - n
    if f < 0:
        f = 0

    bar = "█" * w + "░" * n + "X" * f
    return bar[:BAR_WIDTH]


def pct(v, total):
    if total == 0:
        return 0.0
    return (v / total) * 100


def size_bucket(n, max_n):
    """Return which of the HIST_BUCKETS buckets n falls into."""
    bucket_size = max(1, max_n // HIST_BUCKETS)
    return min(HIST_BUCKETS - 1, n // bucket_size)


def render_histogram(size_stats, max_n):
    lines = []

    for row in range(HIST_HEIGHT, 0, -1):
        y_label = f"{int(row / HIST_HEIGHT * 100):>6}% │"
        bar_row = ""
        for bucket in range(HIST_BUCKETS):
            w, n, f = size_stats[bucket]
            tot = w + n + f
            if tot == 0:
                bar_row += "  "
            else:
                win_rows  = round(w / tot * HIST_HEIGHT)
                norm_rows = round(n / tot * HIST_HEIGHT)
                if row <= win_rows:
                    bar_row += "██"
                elif row <= win_rows + norm_rows:
                    bar_row += "░░"
                else:
                    bar_row += "XX"
        lines.append(y_label + bar_row)

    lines.append("         └" + "─" * (HIST_BUCKETS * 2))

    bucket_size = max_n / HIST_BUCKETS
    label_row = "          "
    tick_row  = "          "
    for i in range(HIST_BUCKETS):
        if i % 5 == 0:
            val = int(i * bucket_size)
            if val == 0:           lbl = "0"
            elif val >= 1_000_000: lbl = f"{val/1_000_000:.1f}M"
            elif val >= 1_000:     lbl = f"{val/1_000:.0f}K"
            else:                  lbl = str(val)
            label_row += f"{lbl:<10}"
            tick_row  += "│         "
    lines.append(tick_row)
    lines.append(label_row)
    lines.append("         W:██  P:░░  F:XX  — win rate by array size")
    return lines


# ============================================================
# DASHBOARD
# ============================================================


LAST_DASHBOARD_HEIGHT = 0


def render_dashboard(start, global_stats, dist_stats, size_stats, total):
    global LAST_DASHBOARD_HEIGHT

    runtime = format_runtime(time.time() - start)
    win, normal, fail = global_stats

    lines = []
    lines.append(f"RUNTIME {runtime} | TESTS {total}")

    bar = make_bar(win, normal, fail, total)
    lines.append(
        f"{'GLOBAL':10} [{bar}] "
        f"W:{win}({pct(win,total):.1f}%) "
        f"P:{normal}({pct(normal,total):.1f}%) "
        f"F:{fail}({pct(fail,total):.1f}%)"
    )

    if ARGS.detailed_mode:
        lines.append("DISTRIBUTIONS")
        for name in dist_stats:
            w, n, f = dist_stats[name]
            tot = w + n + f
            bar = make_bar(w, n, f, tot)
            lines.append(
                f"{name:10} [{bar}] "
                f"W:{w}({pct(w,tot):.1f}%) "
                f"P:{n}({pct(n,tot):.1f}%) "
                f"F:{f}({pct(f,tot):.1f}%)"
            )

    # ── win-rate histogram (opt-in via --size-hist) ───────────
    if ARGS.size_hist:
        lines.append("")
        lines.extend(render_histogram(size_stats, CONFIG["MAX_N"]))

    if LAST_DASHBOARD_HEIGHT > 0:
        sys.stdout.write(f"\033[{LAST_DASHBOARD_HEIGHT}A\r")

    for line in lines:
        sys.stdout.write(line + "\n")
    sys.stdout.flush()

    if LAST_DASHBOARD_HEIGHT > len(lines):
        for _ in range(LAST_DASHBOARD_HEIGHT - len(lines)):
            sys.stdout.write("\033[K\n")
        sys.stdout.write(f"\033[{LAST_DASHBOARD_HEIGHT - len(lines)}A")

    LAST_DASHBOARD_HEIGHT = len(lines)


# ============================================================
# MAIN
# ============================================================


def main():
    print("--- BLACK PROJECT: ARCHITECTURE VALIDATION ---")
    print(f"OS/ARCH:   {platform.system()} {platform.machine()}")
    print(f"W: Faster than Numpy | P: Correct but slower | F: failed to sort")
    if ARGS.distr:
        print(f"DISTR:     {ARGS.distr} only")

    start_time = time.time()

    win = 0
    normal = 0
    fail = 0
    total = 0
    it = 0

    dist_stats = {name: [0, 0, 0] for name, _ in DISTS}
    size_stats = [[0, 0, 0] for _ in range(HIST_BUCKETS)]

    try:
        while True:
            it += 1

            if ARGS.iter and it > ARGS.iter:
                break
            if ARGS.time and (time.time() - start_time) > ARGS.time:
                break

            n = random.randint(0, CONFIG["MAX_N"])
            name, gen = random.choice(ACTIVE_DISTS)

            arr = gen(n)
            arr_copy = arr.copy()

            if random.random() < 0.5:
                # numpy first
                t0 = time.perf_counter()
                ref = np.sort(arr_copy)
                t_ref = (time.perf_counter() - t0) * 1000

                t1 = time.perf_counter()
                try:
                    super_fast_sort(arr)
                    t_fast = (time.perf_counter() - t1) * 1000
                    valid = np.array_equal(arr, ref)
                except Exception as e:
                    print(f"Exception during sorting: {e}")
                    valid = False
                    t_fast = 0
            else:
                # super_fast_sort first
                t1 = time.perf_counter()
                try:
                    super_fast_sort(arr)
                    t_fast = (time.perf_counter() - t1) * 1000
                except Exception as e:
                    print(f"Exception during sorting: {e}")
                    valid = False
                    t_fast = 0
                    # still need ref for correctness check
                    ref = np.sort(arr_copy)
                else:
                    t0 = time.perf_counter()
                    ref = np.sort(arr_copy)
                    t_ref = (time.perf_counter() - t0) * 1000
                    valid = np.array_equal(arr, ref)

            total += 1

            b = size_bucket(n, CONFIG["MAX_N"])
            if not valid:
                fail += 1
                dist_stats[name][2] += 1
                size_stats[b][2] += 1
                if not ARGS.no_fail:
                    print(f"Stopping due to failure on distribution: {name} (n={n})")
                    break
            else:
                speedup = t_ref / t_fast if t_fast > 0 else 0
                if speedup > 1.0:
                    win += 1
                    dist_stats[name][0] += 1
                    size_stats[b][0] += 1
                else:
                    normal += 1
                    dist_stats[name][1] += 1
                    size_stats[b][1] += 1

            render_dashboard(
                start_time,
                (win, normal, fail),
                dist_stats,
                size_stats,
                total,
            )

    except KeyboardInterrupt:
        pass

    print("\nRun finished")


if __name__ == "__main__":
    main()