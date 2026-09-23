"""Summarise a sweep's results over every parameter but the ones named.

    python group_sweep.py                                   # the latest sweep
    python group_sweep.py plots/sweep_002/success.csv
    python group_sweep.py -by effort_weight train_horizon   # the default grouping
    python group_sweep.py -by plot_horizon -o by_ph.csv
    python group_sweep.py -target 0.8                       # what counts as reliable
    python group_sweep.py -pair none                        # skip the paired csv

Writes two csvs beside the input (or wherever -o and -po say):

success_grouped.csv, one row per combination of the grouping parameters, over
every row that shares it -- every plot horizon and every start:

    success_mean/std/min/median  the spread of the success fraction
    success_frac_ge_<target>     how often a run reaches the target
    success_ci95                 half-width of a 95% t-interval on the mean
                                 success across starts; empty with only one,
                                 since the rows then differ only in horizon and
                                 prefixes of one run are not independent draws
    mean_abs_u_mean/std          the effort that bought it
    baseline_mean                success with u = 0 from the same starts, to
                                 the same horizons: a lone start's attractor
                                 can sit well off 0.5 on its own
    gain_mean                    success - baseline, run by run
    success_per_u, gain_per_u    the means above per unit of mean |u|
    on_pareto                    1 if no other group has both more success and
                                 less effort
    success_ph_slope             least-squares success per Lyapunov time of
                                 plot horizon: below zero, the control is
                                 losing its grip as the run goes on
    success_at_max_ph            mean success at the longest plot horizon

success_paired.csv, one row per value of the pair parameter (default
train_horizon) against its lowest value, matched run for run on everything else
-- the same start, plot horizon and remaining grouping values -- so the start
and the horizon cancel out of the difference:

    success_diff_mean, mean_abs_u_diff_mean
    success_diff_frac_pos        how often the higher value did better
    success_diff_ci95            across starts, as above

The baseline needs lorenz.py, and with it numpy and torch; everything else is
the standard library.
"""

import argparse
import csv
import glob
import os
import statistics
from collections import defaultdict

SWEEP_ROOT = "plots"
# what a sweep can have measured, in params.RESULT_COLUMNS order; a csv from
# before a column existed simply has fewer of them
RESULT_COLUMNS = ("success", "mean_abs_u")
# two-sided 95% Student t critical values for 1..30 degrees of freedom
T95 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)  # fmt: skip


def latest_sweep_csv():
    found = glob.glob(os.path.join(SWEEP_ROOT, "sweep_*", "success.csv"))

    if not found:
        raise SystemExit(f"no sweeps found under {SWEEP_ROOT}")

    return max(found, key=os.path.getmtime)


def value(text):
    # parameters sort and print as numbers where they are numbers; the start,
    # written x|y|z, stays text
    try:
        return float(text)
    except ValueError:
        return text


def mean(v):
    return statistics.mean(v) if v else None


def ci95(v):
    if len(v) < 2:
        return None

    t = T95[len(v) - 2] if len(v) - 1 <= len(T95) else 1.96

    return t * statistics.stdev(v) / len(v) ** 0.5


def slope(xs, ys):
    if len(set(xs)) < 2:
        return None

    x0, y0 = statistics.mean(xs), statistics.mean(ys)

    return sum((x - x0) * (y - y0) for x, y in zip(xs, ys)) / sum(
        (x - x0) ** 2 for x in xs
    )


def per_start_means(rows, column):
    # a start's runs at different horizons are prefixes of one trajectory, so
    # the start, not the row, is the independent draw an interval is taken over
    starts = defaultdict(list)

    for row in rows:
        starts[row["initial_condition"]].append(row[column])

    return [statistics.mean(v) for v in starts.values()]


def attach_baseline(rows):
    """Give each row the success an uncontrolled run from its start would have had.

    Counted over exactly the steps the controlled run was scored on:
    training.horizon_steps tiles the horizon in whole windows, so the same plot
    horizon is a slightly different number of steps for each training window.
    One rollout per start and integrator covers every horizon, read off as
    prefixes of it.
    """
    # imported here so a csv without the columns doesn't pay for torch
    import numpy as np

    from lorenz import DT, LYAPUNOV_EXP, euler_step, rk4_step, rollout_numpy_batched
    from training import horizon_steps

    unit = LYAPUNOV_EXP * DT

    for row in rows:
        th, ph = row["train_horizon"], row["plot_horizon"]
        row["_steps"] = horizon_steps(ph, th, round(th / unit))

    by_integrator = defaultdict(list)

    for row in rows:
        by_integrator[row.get("rk4", 1.0)].append(row)

    for rk4, group in by_integrator.items():
        starts = sorted({row["initial_condition"] for row in group})
        state0 = np.array([[float(c) for c in s.split("|")] for s in starts])
        steps = max(row["_steps"] for row in group)

        traj, _ = rollout_numpy_batched(
            state0,
            np.zeros_like(state0),
            np.zeros(len(starts)),
            steps=steps,
            integrator=rk4_step if rk4 else euler_step,
        )
        # the success of the first n+1 states, for every n at once
        ahead = np.cumsum(traj[..., 0] > 0, axis=0) / np.arange(1, steps + 2)[:, None]
        lane = {start: i for i, start in enumerate(starts)}

        for row in group:
            row["baseline"] = float(ahead[row["_steps"], lane[row["initial_condition"]]])
            row["gain"] = row["success"] - row["baseline"]


def read(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = []

        for raw in reader:
            # a point that failed to run leaves its result cells empty
            if not raw.get("success"):
                continue

            row = {name: value(raw[name]) for name in header}

            for name in RESULT_COLUMNS:
                if name in row:
                    row[name] = float(raw[name]) if raw[name] else None

            rows.append(row)

    return header, rows


def fmt(x):
    if x is None:
        return ""

    return f"{x:.6f}" if isinstance(x, float) else x


def label(x):
    # a parameter prints as the sweep wrote it, 0.5 rather than 0.500000
    return f"{x:g}" if isinstance(x, float) else x


def write(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows([fmt(row.get(name)) for name in header] for row in rows)


def summarise(rows, by, target, has_u, has_baseline):
    groups = defaultdict(list)

    for row in rows:
        groups[tuple(row[name] for name in by)].append(row)

    out = []

    for key in sorted(groups):
        group = groups[key]
        success = [row["success"] for row in group]
        s = {name: label(x) for name, x in zip(by, key)}

        s["success_mean"] = mean(success)
        s["success_std"] = statistics.stdev(success) if len(success) > 1 else None
        s["success_min"] = min(success)
        s["success_median"] = statistics.median(success)
        s["success_frac_ge"] = sum(x >= target for x in success) / len(success)
        s["success_ci95"] = ci95(per_start_means(group, "success"))

        if has_u:
            u = [row["mean_abs_u"] for row in group if row["mean_abs_u"] is not None]
            s["mean_abs_u_mean"] = mean(u)
            s["mean_abs_u_std"] = statistics.stdev(u) if len(u) > 1 else None
            s["success_per_u"] = s["success_mean"] / s["mean_abs_u_mean"]

        if has_baseline:
            s["baseline_mean"] = mean([row["baseline"] for row in group])
            s["gain_mean"] = mean([row["gain"] for row in group])

            if has_u:
                s["gain_per_u"] = s["gain_mean"] / s["mean_abs_u_mean"]

        if "plot_horizon" in group[0] and "plot_horizon" not in by:
            longest = max(row["plot_horizon"] for row in group)
            s["success_ph_slope"] = slope(
                [row["plot_horizon"] for row in group], success
            )
            s["success_at_max_ph"] = mean(
                [row["success"] for row in group if row["plot_horizon"] == longest]
            )

        s["n_starts"] = len({row.get("initial_condition") for row in group})
        s["count"] = len(group)
        out.append(s)

    if has_u:
        for s in out:
            s["on_pareto"] = int(
                not any(
                    o["success_mean"] >= s["success_mean"]
                    and o["mean_abs_u_mean"] <= s["mean_abs_u_mean"]
                    and (o["success_mean"], o["mean_abs_u_mean"])
                    != (s["success_mean"], s["mean_abs_u_mean"])
                    for o in out
                )
            )

    return out


def pair(rows, params, by, column, has_u):
    """Each value of `column` against its lowest, matched run for run."""
    reference = min(row[column] for row in rows)
    rest = [name for name in params if name != column]
    matched = {tuple(row[name] for name in rest): row for row in rows if row[column] == reference}
    others = [name for name in by if name != column]
    groups = defaultdict(list)

    for row in rows:
        partner = matched.get(tuple(row[name] for name in rest))

        if row[column] != reference and partner is not None:
            key = (*(row[name] for name in others), row[column])
            groups[key].append((row, partner))

    out = []

    for key in sorted(groups):
        runs = groups[key]
        diffs = [a["success"] - b["success"] for a, b in runs]
        s = {name: label(x) for name, x in zip(others, key[:-1])}

        s[column], s[f"{column}_ref"] = label(key[-1]), label(reference)
        s["success_diff_mean"] = mean(diffs)
        s["success_diff_frac_pos"] = sum(d > 0 for d in diffs) / len(diffs)
        s["success_diff_ci95"] = ci95(
            per_start_means(
                [{"initial_condition": a["initial_condition"], "d": d} for (a, _), d in zip(runs, diffs)],
                "d",
            )
        )

        if has_u:
            s["mean_abs_u_diff_mean"] = mean(
                [a["mean_abs_u"] - b["mean_abs_u"] for a, b in runs]
            )

        s["count"] = len(runs)
        out.append(s)

    return out, others


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "csv", nargs="?", help="a sweep's success.csv (default: the latest sweep)"
    )
    parser.add_argument(
        "-by",
        nargs="+",
        default=["effort_weight", "train_horizon"],
        help="the parameters to group by (default: effort_weight train_horizon)",
    )
    parser.add_argument(
        "-target",
        type=float,
        default=0.7,
        help="the success a run must reach to count as reliable (default: 0.7)",
    )
    parser.add_argument(
        "-pair",
        default="train_horizon",
        help="the parameter to difference run for run, or none (default: train_horizon)",
    )
    parser.add_argument(
        "-o", "--out", help="where to write (default: success_grouped.csv beside the input)"
    )
    parser.add_argument(
        "-po",
        "--pair_out",
        help="where to write the paired csv (default: success_paired.csv beside the input)",
    )
    args = parser.parse_args()

    path = args.csv or latest_sweep_csv()
    header, rows = read(path)
    params = [name for name in header if name not in RESULT_COLUMNS]
    missing = [name for name in args.by if name not in params]

    if missing:
        raise SystemExit(f"{path} has no column {', '.join(missing)}")

    has_u = "mean_abs_u" in header and all(row["mean_abs_u"] is not None for row in rows)
    has_baseline = {"initial_condition", "train_horizon", "plot_horizon"} <= set(params)

    if has_baseline:
        attach_baseline(rows)

    grouped = summarise(rows, args.by, args.target, has_u, has_baseline)
    frac = f"success_frac_ge_{args.target:g}"
    columns = [
        *args.by,
        "success_mean",
        "success_std",
        "success_min",
        "success_median",
        frac,
        "success_ci95",
        *(["mean_abs_u_mean", "mean_abs_u_std"] if has_u else []),
        *(["baseline_mean", "gain_mean"] if has_baseline else []),
        *(["success_per_u"] if has_u else []),
        *(["gain_per_u"] if has_u and has_baseline else []),
        *(["on_pareto"] if has_u else []),
        *(["success_ph_slope", "success_at_max_ph"] if "success_ph_slope" in grouped[0] else []),
        "n_starts",
        "count",
    ]

    for s in grouped:
        s[frac] = s.pop("success_frac_ge")

    out = args.out or os.path.join(os.path.dirname(path), "success_grouped.csv")
    write(out, columns, grouped)
    print(f"{len(grouped)} groups of {', '.join(args.by)} -> {out}")

    if args.pair == "none":
        return

    if args.pair not in params or len({row[args.pair] for row in rows}) < 2:
        print(f"{args.pair} has fewer than two values; no paired csv")
        return

    paired, others = pair(rows, params, args.by, args.pair, has_u)
    pair_out = args.pair_out or os.path.join(os.path.dirname(path), "success_paired.csv")
    write(
        pair_out,
        [
            *others,
            args.pair,
            f"{args.pair}_ref",
            "success_diff_mean",
            "success_diff_frac_pos",
            "success_diff_ci95",
            *(["mean_abs_u_diff_mean"] if has_u else []),
            "count",
        ],
        paired,
    )
    print(f"{len(paired)} pairs of {args.pair} -> {pair_out}")


if __name__ == "__main__":
    main()
