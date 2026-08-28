"""Sweep run_control.py over a grid of any run parameters.

Each grid point runs as its own run_control.py subprocess, so the sweep is
parallelized simply by running several of them at once.

    python sweep.py                             # the default lambda x window grid
    python sweep.py -sw lr=0.01:0.1:0.01        # sweep the learning rate instead
    python sweep.py -sw lam=0.05:0.15:0.01 -sw th=1:2:0.25 -sw rk4=0,1
    python sweep.py -j 4                        # cap at 4 concurrent runs
    python sweep.py -loss                       # also save a loss curve per point
    python sweep.py -np                         # skip the success plot at the end
    python sweep.py -sc                         # just replot the latest sweep
    python sweep.py --dry-run                   # print the commands, run nothing

An axis is NAME=start:stop:step (inclusive of stop) or NAME=v1,v2,v3, where
NAME is any sweepable parameter in params.py, by canonical name or short key.
Anything not swept keeps its default unless the matching flag below sets it.

Each sweep claims the next number from plots/.sweep_counter and writes
everything it produces into that one directory:

    plots/sweep_007/
        manifest.json               # the axes, the constants, the git sha
        success.csv                 # one row per grid point, every parameter
        success_v_lam.png           # unless -np
        attractor/                  # one png per grid point
        loss_v_iteration/           # one png per grid point, with -loss

Filenames carry only the axes the sweep varied — `lam0p100_th1p000.png` — so a
constant never repeats across every name in the directory; the manifest and the
csv hold the rest. Numbers are never reused, so a directory name stays a stable
reference to one run. -sc with no argument replots the most recent one.
"""

import argparse
import csv
import glob
import itertools
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from params import PARAMS, SWEEPABLE, csv_value, defaults, parse_axis, resolve, stem

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_control.py")
SUCCESS_RE = re.compile(r"success \(fraction of time x > 0\): ([0-9.]+)")

SWEEP_ROOT = "./plots"
COUNTER = os.path.join(SWEEP_ROOT, ".sweep_counter")

# what `python sweep.py` with no axes has always done
DEFAULT_AXES = ("effort_weight=0.05:0.15:0.01", "train_horizon=1.0:2.0:0.25")


def claim_sweep_dir():
    """Create the next numbered directory under plots/ and return its path.

    The counter file is what makes the numbering monotonic: deleting old sweeps
    doesn't hand their numbers out again, so a directory name stays a stable
    reference to one run. mkdir is the actual claim, so two sweeps started at
    the same moment can't both take the same number.
    """
    os.makedirs(SWEEP_ROOT, exist_ok=True)

    try:
        with open(COUNTER) as f:
            n = int(f.read().strip())
    except (FileNotFoundError, ValueError):
        n = 0

    while True:
        n += 1
        path = os.path.join(SWEEP_ROOT, f"sweep_{n:03d}")

        try:
            os.mkdir(path)
            break
        except FileExistsError:
            continue

    with open(COUNTER, "w") as f:
        f.write(str(n))

    return path


def latest_sweep_csv():
    found = glob.glob(os.path.join(SWEEP_ROOT, "sweep_*", "success.csv"))

    if not found:
        raise SystemExit(f"no sweeps found under {SWEEP_ROOT}")

    return max(found, key=os.path.getmtime)


def plot_csv(path, save, x_key=None, panel_key=None):
    # imported here so a plain sweep doesn't pay for torch and matplotlib
    from figures import plot_success

    plot_success(path, x_key=x_key, panel_key=panel_key, save=save)


def git_sha():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(SCRIPT),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def build_axes(specs):
    axes = {}

    for spec in specs:
        name, values = parse_axis(spec)

        if name in axes:
            raise SystemExit(f"{name} is swept twice")

        axes[name] = values

    return axes


def build_base(args, axes):
    """The constant parameters: defaults, then whatever the flags overrode."""
    overrides = {
        name: value
        for name, value in (
            ("initial_condition", args.initial_condition),
            ("learning_rate", args.learning_rate),
            ("plot_horizon", args.plot_horizon),
            ("iters", args.iters),
        )
        if value is not None
    }

    # a store_true flag can't tell "not given" from "off", so only a set one
    # counts as an override
    if args.no_penalize_effort:
        overrides["penalize_effort"] = False
    if args.rk4:
        overrides["rk4"] = True

    clash = sorted(set(overrides) & set(axes))

    if clash:
        raise SystemExit(
            f"{', '.join(clash)} is both swept and set by a flag; drop one"
        )

    base = defaults()
    # the sweep has always regularized unless told otherwise
    base["penalize_effort"] = True
    base.update(overrides)

    return base


def build_command(values, sweep_dir, name_keys, loss_curve):
    cmd = [
        sys.executable,
        SCRIPT,
        "-o",
        sweep_dir,
        "-nk",
        ",".join(name_keys),
        "-s",
    ]

    for param in PARAMS:
        value = values[param.name]

        if param.store_true:
            if value:
                cmd.append(param.flag)
        elif param.nargs > 1:
            cmd += [param.flag, *(str(component) for component in value)]
        else:
            cmd += [param.flag, str(value)]

    if loss_curve:
        cmd.append("-loss")

    return cmd


def run(cmd, env):
    start = time.monotonic()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return proc, time.monotonic() - start


def parse_success(stdout):
    match = SUCCESS_RE.search(stdout)

    return float(match.group(1)) if match else None


def write_csv(path, rows, axis_names):
    """One row per grid point, every parameter as its own column.

    The constants are as redundant here as they'd be in a filename, but a csv
    that carries them stays self-describing once it's been moved away from its
    sweep, and it lets the plotter work out on its own what varied.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    columns = [param.name for param in PARAMS]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([*columns, "success"])

        for values, success in sorted(
            rows, key=lambda row: [row[0][name] for name in axis_names]
        ):
            writer.writerow(
                [
                    *(csv_value(name, values[name]) for name in columns),
                    "" if success is None else f"{success:.6f}",
                ]
            )

    print(f"wrote {len(rows)} rows to {path}")


def write_manifest(path, axes, base, name_keys, args, grid_size):
    """Everything the filenames deliberately leave out."""
    constant = {
        name: csv_value(name, value)
        for name, value in base.items()
        if name not in axes
    }

    with open(path, "w") as f:
        json.dump(
            {
                "sweep": os.path.basename(os.path.dirname(os.path.abspath(path))),
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "git_sha": git_sha(),
                "command": " ".join(sys.argv),
                "axes": axes,
                "name_keys": name_keys,
                "constant": constant,
                "grid_points": grid_size,
                "jobs": args.jobs,
            },
            f,
            indent=2,
            sort_keys=False,
        )
        f.write("\n")

    print(f"wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-sw",
        "--sweep",
        action="append",
        metavar="NAME=SPEC",
        help="an axis: NAME=start:stop:step or NAME=v1,v2,v3, repeatable. "
        f"sweepable parameters: {', '.join(SWEEPABLE)}",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=os.cpu_count(),
        help="max concurrent run_control.py runs (default: all cores)",
    )

    # passed straight through to run_control.py; None means "leave at default",
    # which is what lets a clash with a swept axis be caught
    parser.add_argument(
        "-ic",
        "--initial_condition",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("-lr", "--learning_rate", type=float, default=None)
    parser.add_argument("-ph", "--plot_horizon", type=float, default=None)
    parser.add_argument("-i", "--iters", type=int, default=None)

    parser.add_argument(
        "--csv",
        default=None,
        help="where to write the success rates (default: <sweep dir>/success.csv)",
    )
    parser.add_argument(
        "-sc",
        "--success_csv",
        nargs="?",
        const="",
        default=None,
        help="plot an existing csv instead of sweeping (default: the latest sweep)",
    )
    parser.add_argument(
        "-x",
        "--x_key",
        default=None,
        help="parameter on the success plot's x axis (default: the first axis swept)",
    )
    parser.add_argument(
        "-p",
        "--panel_key",
        default=None,
        help="parameter to give one success panel per value (default: the second "
        "axis swept)",
    )

    flags = parser.add_argument_group(title="Flags")
    flags.add_argument("-s", "--save", action="store_true")
    flags.add_argument(
        "-np",
        "--no_plot",
        action="store_true",
        help="skip the success plot the grid otherwise saves alongside its csv",
    )
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

    # fail on a typo'd axis name now, not after the whole grid has run
    for key in (args.x_key, args.panel_key):
        if key:
            resolve(key)

    if args.success_csv is not None:
        plot_csv(
            args.success_csv or latest_sweep_csv(),
            save=args.save,
            x_key=args.x_key,
            panel_key=args.panel_key,
        )
        sys.exit(0)

    axes = build_axes(args.sweep or DEFAULT_AXES)
    base = build_base(args, axes)

    axis_names = list(axes)
    grid = [
        {**base, **dict(zip(axis_names, combination))}
        for combination in itertools.product(*axes.values())
    ]

    # only what actually varies goes in a filename; a one-value axis is a
    # constant with extra steps, and belongs in the manifest like the rest
    name_keys = [name for name in axis_names if len(set(axes[name])) > 1] or axis_names

    for name, values in axes.items():
        print(f"{name}: {values}")

    print(f"{len(grid)} runs, {args.jobs} at a time")
    print(f"naming by: {', '.join(name_keys)}")

    # the axes decide the names, so a name can only collide if a formatter is
    # coarser than its axis's step. cheaper to find here than after the grid runs
    stems = [stem(values, name_keys) for values in grid]
    collisions = sorted({s for s in stems if stems.count(s) > 1})

    if collisions:
        raise SystemExit(
            f"{len(collisions)} filename collision(s), e.g. {collisions[0]}; "
            f"give the offending parameter more decimal places in params.py"
        )

    if args.dry_run:
        # a dry run claims no number, so the directory here is a placeholder
        for values in grid:
            print(
                " ".join(
                    build_command(
                        values,
                        f"{SWEEP_ROOT}/sweep_NNN",
                        name_keys,
                        args.loss_curve,
                    )
                )
            )
        sys.exit(0)

    sweep_dir = claim_sweep_dir()
    csv_path = args.csv or os.path.join(sweep_dir, "success.csv")
    print(f"writing to {sweep_dir}")

    write_manifest(
        os.path.join(sweep_dir, "manifest.json"),
        axes,
        base,
        name_keys,
        args,
        len(grid),
    )

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
            pool.submit(
                run,
                build_command(values, sweep_dir, name_keys, args.loss_curve),
                env,
            ): values
            for values in grid
        }

        for done, future in enumerate(as_completed(futures), start=1):
            values = futures[future]
            proc, elapsed = future.result()
            status = "ok " if proc.returncode == 0 else "FAIL"
            success = parse_success(proc.stdout) if proc.returncode == 0 else None
            rows.append((values, success))
            where = "  ".join(f"{name}={values[name]}" for name in axis_names)

            print(
                f"[{done}/{len(grid)}] {status} {where}  {elapsed:.1f}s"
                f"  success={'--' if success is None else f'{success:.4f}'}",
                flush=True,
            )

            if proc.returncode != 0:
                failures.append((where, proc.stderr.strip()))

    print(f"\nfinished in {time.monotonic() - started:.1f}s")

    write_csv(csv_path, rows, axis_names)

    # a finished sweep is usually headless, so write the figure rather than
    # blocking on a window nobody is watching. an all-failed grid has nothing to
    # plot, and raising over it would bury the failure report below.
    if not args.no_plot and any(success is not None for _, success in rows):
        plot_csv(csv_path, save=True, x_key=args.x_key, panel_key=args.panel_key)

    missing = [values for values, success in rows if success is None]

    if missing:
        print(f"{len(missing)} of {len(rows)} runs reported no success rate")

    for where, stderr in failures:
        print(f"\n{where} failed:\n{stderr}")

    sys.exit(1 if failures else 0)
