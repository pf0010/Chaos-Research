"""Sweep control_plots.py over a grid of lambda and train_lyapunov_time values.

Each grid point runs as its own control_plots.py subprocess, so the sweep is
parallelized simply by running several of them at once.

    python sweep.py                 # full grid, one worker per core
    python sweep.py -j 4            # cap at 4 concurrent runs
    python sweep.py -loss           # also save a loss-vs-iteration plot per point
    python sweep.py --dry-run       # print the commands without running them

Every run also writes the success rate of each grid point to --csv.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "control_plots.py")
SUCCESS_RE = re.compile(r"success \(fraction of time x > 0\): ([0-9.]+)")


def frange(start, stop, step):
    # inclusive of stop, with a tolerance so 0.15 isn't dropped by float error
    values = []
    n = 0
    while True:
        value = round(start + n * step, 6)
        if value > stop + step / 2:
            break
        values.append(value)
        n += 1
    return values


def build_command(lam, train_horizon, args):
    cmd = [
        sys.executable,
        SCRIPT,
        "-lam",
        str(lam),
        "-th",
        str(train_horizon),
        "-lr",
        str(args.learning_rate),
        "-ph",
        str(args.plot_horizon),
        "-i",
        str(args.iters),
        "-ic",
        *(str(v) for v in args.initial_condition),
        "-s",
    ]

    if not args.no_penalize_effort:
        cmd.append("-pe")
    if args.rk4:
        cmd.append("-rk4")
    if args.loss_curve:
        cmd.append("-loss")

    return cmd


def run(cmd, env):
    start = time.monotonic()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return proc, time.monotonic() - start


def parse_success(stdout):
    match = SUCCESS_RE.search(stdout)

    return float(match.group(1)) if match else None


def write_csv(path, rows, args):
    # the run settings are constant across the grid, but carrying them along
    # keeps a csv self-describing once it's been moved away from the sweep
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "effort_weight",
                "train_horizon",
                "success",
                "learning_rate",
                "iters",
                "plot_horizon",
                "penalize_effort",
                "rk4",
            ]
        )

        for lam, train_horizon, success in sorted(rows):
            writer.writerow(
                [
                    lam,
                    train_horizon,
                    "" if success is None else f"{success:.6f}",
                    args.learning_rate,
                    args.iters,
                    args.plot_horizon,
                    int(not args.no_penalize_effort),
                    int(args.rk4),
                ]
            )

    print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lam_start", type=float, default=0.05)
    parser.add_argument("--lam_stop", type=float, default=0.15)
    parser.add_argument("--lam_step", type=float, default=0.01)
    parser.add_argument("--th_start", type=float, default=1.0)
    parser.add_argument("--th_stop", type=float, default=2.0)
    parser.add_argument("--th_step", type=float, default=0.25)

    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=os.cpu_count(),
        help="max concurrent control_plots.py runs (default: all cores)",
    )

    # passed straight through to control_plots.py
    parser.add_argument(
        "-ic",
        "--initial_condition",
        nargs=3,
        type=float,
        default=[0, 1, 1.05],
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.05)
    parser.add_argument("-ph", "--plot_horizon", type=float, default=100)
    parser.add_argument("-i", "--iters", type=int, default=600)

    parser.add_argument(
        "--csv",
        default="./plots/loss_sweep/success.csv",
        help="where to write the success rates (default: %(default)s)",
    )

    flags = parser.add_argument_group(title="Flags")
    flags.add_argument("-rk4", "--rk4", action="store_true")
    flags.add_argument(
        "-loss",
        "--loss_curve",
        action="store_true",
        help="also save a loss-vs-iteration plot for every grid point",
    )
    flags.add_argument(
        "-npe",
        "--no_penalize_effort",
        action="store_true",
        help="drop -pe; lambda then has no effect on training",
    )
    flags.add_argument("--dry_run", "--dry-run", action="store_true", dest="dry_run")
    args = parser.parse_args()

    lams = frange(args.lam_start, args.lam_stop, args.lam_step)
    train_horizons = frange(args.th_start, args.th_stop, args.th_step)
    grid = [(lam, th) for lam in lams for th in train_horizons]

    print(f"effort_weight: {lams}")
    print(f"train_horizon: {train_horizons}")
    print(f"{len(grid)} runs, {args.jobs} at a time")

    if args.dry_run:
        for lam, th in grid:
            print(" ".join(build_command(lam, th, args)))
        sys.exit(0)

    # each run is single-threaded so the workers don't oversubscribe the CPU,
    # and Agg keeps matplotlib from reaching for a GUI backend
    env = {
        **os.environ,
        "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }

    failures = []
    rows = []
    started = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(run, build_command(lam, th, args), env): (lam, th)
            for lam, th in grid
        }

        for done, future in enumerate(as_completed(futures), start=1):
            lam, th = futures[future]
            proc, elapsed = future.result()
            status = "ok " if proc.returncode == 0 else "FAIL"
            success = parse_success(proc.stdout) if proc.returncode == 0 else None
            rows.append((lam, th, success))

            print(
                f"[{done}/{len(grid)}] {status} effort_weight={lam} "
                f"train_horizon={th}  {elapsed:.1f}s"
                f"  success={'--' if success is None else f'{success:.4f}'}",
                flush=True,
            )

            if proc.returncode != 0:
                failures.append((lam, th, proc.stderr.strip()))

    print(f"\nfinished in {time.monotonic() - started:.1f}s")

    write_csv(args.csv, rows, args)

    missing = [(lam, th) for lam, th, s in rows if s is None]
    if missing:
        print(f"{len(missing)} of {len(rows)} runs reported no success rate")

    for lam, th, stderr in failures:
        print(f"\neffort_weight={lam} train_horizon={th} failed:\n{stderr}")

    sys.exit(1 if failures else 0)
