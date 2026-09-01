"""Sweep the controller over a grid of any run parameters.

The grid is trained in batches, not one point at a time: λ and the learning
rate ride along a leading batch axis, so a whole row of the grid costs one set
of kernels rather than one set each. Everything that would change the shape of
the computation -- the window length, the integrator, the iteration count --
splits the grid into groups that are trained one after another (params.py marks
which is which). On a GPU the batch is close to free: a group of 11 costs about
what a group of 1 does, because the cost is kernel launches and there are the
same number either way. See kernels.py.

    python sweep.py                             # the default lambda x window grid
    python sweep.py -sw lr=0.01:0.1:0.01        # sweep the learning rate instead
    python sweep.py -sw lam=0.05:0.15:0.01 -sw th=1:2:0.25 -sw rk4=0,1
    python sweep.py -j 4                        # cap at 4 figure-drawing workers
    python sweep.py -loss                       # also save a loss curve per point
    python sweep.py -np                         # skip the success plot at the end
    python sweep.py -sc                         # just replot the latest sweep
    python sweep.py --device cpu                # or --fp64, or --no-graph
    python sweep.py --dry-run                   # print the plan, run nothing

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
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

from params import (
    PARAMS,
    SHAPING,
    SWEEPABLE,
    csv_value,
    defaults,
    parse_axis,
    resolve,
    stem,
)

HERE = os.path.dirname(os.path.abspath(__file__))

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
            cwd=HERE,
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
    if args.euler:
        overrides["rk4"] = False

    clash = sorted(set(overrides) & set(axes))

    if clash:
        raise SystemExit(
            f"{', '.join(clash)} is both swept and set by a flag; drop one"
        )

    base = defaults()
    base.update(overrides)

    return base


def group_grid(grid):
    """Split the grid into batches that can train together.

    A group is a set of points agreeing on everything except the parameters
    params.py marks batchable, so within a group only per-element numbers
    differ and one run of the training loop covers all of them.
    """
    groups = {}

    for values in grid:
        key = tuple(csv_value(name, values[name]) for name in SHAPING)
        # a point whose effort window happens to coincide with its training
        # window is scored by the pathwise penalty rather than the frozen
        # moments -- a different objective, so it gets its own group
        key += (values["plot_horizon"] == values["train_horizon"],)
        groups.setdefault(key, []).append(values)

    return list(groups.values())


def train_group(group, device, dtype, use_graph):
    """Train one batch and evaluate it, returning (w, b, history, traj, u, success).

    The evaluation rollout is batched too, and it is the one the figures and
    the success metric are read off, so every point in the group is measured
    on exactly the trajectory that is drawn for it.
    """
    # imported here so --dry-run and -sc don't pay for torch and matplotlib
    from lorenz import DT, LYAPUNOV_EXP, euler_step, rk4_step, rollout_numpy_batched
    from training import success_fraction, train_policy_batched

    settings = group[0]
    integrator = rk4_step if settings["rk4"] else euler_step

    w, b, history = train_policy_batched(
        state0=settings["initial_condition"],
        batch=len(group),
        horizon=[values["train_horizon"] for values in group],
        effort_horizon=settings["plot_horizon"],
        iters=settings["iters"],
        learning_rate=[values["learning_rate"] for values in group],
        effort_weight=[values["effort_weight"] for values in group],
        penalize_effort=settings["penalize_effort"],
        integrator=integrator,
        device=device,
        dtype=dtype,
        use_graph=use_graph,
    )

    w = w.cpu().double().numpy()
    b = b.cpu().double().numpy()

    steps = round(settings["plot_horizon"] / (LYAPUNOV_EXP * DT))
    traj, u = rollout_numpy_batched(
        settings["initial_condition"], w, b, steps=steps, integrator=integrator
    )

    return w, b, history, traj, u, success_fraction(traj)


def draw_point(job):
    """One grid point's figures, in a worker process. Returns its filename stem."""
    import contextlib
    import io

    from figures import plot_loss_curve, plot_run_summary

    values, w, b, history, traj, u, sweep_dir, name_keys, loss_curve = job

    import torch

    params = (torch.as_tensor(w), torch.as_tensor(b))
    shared = dict(
        state0=values["initial_condition"],
        lr=values["learning_rate"],
        train_horizon=values["train_horizon"],
        plot_horizon=values["plot_horizon"],
        iters=values["iters"],
        penalize_effort=values["penalize_effort"],
        effort_weight=values["effort_weight"],
        integrator=rk4_step_if(values["rk4"]),
        save=True,
        out_dir=sweep_dir,
        name_keys=name_keys,
    )

    # the plotters narrate to stdout, which from 100 workers is noise; the
    # caller prints one line per point instead
    with contextlib.redirect_stdout(io.StringIO()):
        if loss_curve:
            plot_loss_curve(history, **shared)

        plot_run_summary(params, traj=traj, u=u, **shared)

    return stem(values, name_keys)


def rk4_step_if(flag):
    from lorenz import euler_step, rk4_step

    return rk4_step if flag else euler_step


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


def write_manifest(path, axes, base, name_keys, args, grid_size, runtime):
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
                # the device and the precision don't name a file, but they do
                # move the last digits of a result, so the sweep records them
                **runtime,
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
        help="max concurrent figure-drawing workers (default: all cores). The "
        "training itself is batched, not forked, so this no longer sets it",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="where the training runs (default: cuda when there is one)",
    )
    parser.add_argument(
        "--fp64",
        action="store_true",
        help="force float64. The default is float32 on the GPU, where fp64 is "
        "1/64 rate and buys less accuracy than the attractor already destroys",
    )
    parser.add_argument(
        "--no_graph",
        "--no-graph",
        dest="no_graph",
        action="store_true",
        help="skip CUDA graph capture (it is what makes the launch overhead go "
        "away; turn it off to compare, or if capture upsets a driver)",
    )

    # None means "leave at default", which is what lets a clash with a swept
    # axis be caught
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
    flags.add_argument(
        "-euler",
        "--euler",
        action="store_true",
        help="integrate with forward euler instead of the default rk4",
    )
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
        help="train on the task term alone; lambda then has no effect on training",
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

    groups = group_grid(grid)
    batched_over = [name for name in axis_names if name not in SHAPING]

    print(
        f"{len(grid)} runs in {len(groups)} batched group(s) of up to "
        f"{max(len(g) for g in groups)}"
        + (f", batching over {', '.join(batched_over)}" if batched_over else "")
    )
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
        for n, group in enumerate(groups, start=1):
            held = {
                name: group[0][name] for name in SHAPING if name in axis_names
            }
            # the distinct values, not one entry per lane: a merged group holds
            # the whole grid and listing it lane by lane says nothing
            varies = {
                name: sorted({values[name] for values in group})
                for name in batched_over
            }
            print(f"group {n}: {len(group)} point(s)  {held}  batched {varies}")
        sys.exit(0)

    # matplotlib must not reach for a GUI backend, and the drawing workers are
    # forked from here, so this has to be set before figures is ever imported
    os.environ.setdefault("MPLBACKEND", "Agg")

    sweep_dir = claim_sweep_dir()
    csv_path = args.csv or os.path.join(sweep_dir, "success.csv")
    print(f"writing to {sweep_dir}")

    from kernels import default_dtype, resolve_device

    device = resolve_device(args.device)
    import torch

    dtype = torch.float64 if args.fp64 else default_dtype(device)
    use_graph = not args.no_graph

    print(
        f"training on {device} in {str(dtype).split('.')[-1]}"
        + (", CUDA graph captured" if use_graph and device == "cuda" else "")
    )

    write_manifest(
        os.path.join(sweep_dir, "manifest.json"),
        axes,
        base,
        name_keys,
        args,
        len(grid),
        {
            "device": device,
            "dtype": str(dtype).split(".")[-1],
            "cuda_graph": use_graph and device == "cuda",
        },
    )

    rows = []
    jobs = []
    started = time.monotonic()

    for n, group in enumerate(groups, start=1):
        at = time.monotonic()
        w, b, history, traj, u, success = train_group(group, device, dtype, use_graph)
        elapsed = time.monotonic() - at

        held = "  ".join(
            f"{name}={group[0][name]}" for name in SHAPING if name in axis_names
        )
        print(
            f"[group {n}/{len(groups)}] {len(group)} point(s)  "
            + (f"{held}  " if held else "")
            + f"{elapsed:.1f}s  success {success.min():.4f}-{success.max():.4f}",
            flush=True,
        )

        for i, values in enumerate(group):
            rows.append((values, float(success[i])))
            jobs.append(
                (
                    values,
                    w[i],
                    b[i],
                    history[:, i, :],
                    traj[:, i, :],
                    u[:, i],
                    sweep_dir,
                    name_keys,
                    args.loss_curve,
                )
            )

    trained = time.monotonic() - started
    print(f"\ntrained {len(grid)} points in {trained:.1f}s")

    # drawing is the serial part now, so it gets the cores the training used to
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(draw_point, job) for job in jobs]

        for done, future in enumerate(as_completed(futures), start=1):
            print(f"[{done}/{len(jobs)}] drew {future.result()}", flush=True)

    print(f"finished in {time.monotonic() - started:.1f}s")

    write_csv(csv_path, rows, axis_names)

    # a finished sweep is usually headless, so write the figure rather than
    # blocking on a window nobody is watching
    if not args.no_plot:
        plot_csv(csv_path, save=True, x_key=args.x_key, panel_key=args.panel_key)
