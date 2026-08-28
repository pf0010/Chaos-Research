"""Every figure the project draws, for both the free and the controlled system.

Nothing here parses arguments; the run_* modules compose these into a run.
"""

import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from lorenz import (
    DT,
    LYAPUNOV_EXP,
    euler_step,
    rk4_step,
    rollout_numpy,
    rollout_torch,
)
from params import (
    BY_NAME,
    DEFAULT_EFFORT_WEIGHT,
    SWEEPABLE,
    decimal_tag,
    resolve,
    run_values,
    shown,
    stem,
)
from training import (
    SOFTNESS,
    final_state_sensitivity,
    init_policy_params,
    linear_policy,
    soft_step,
    success_fraction,
    task_loss,
)


def run_output_dir(iters, out_dir=None):
    # a sweep hands every grid point its own directory; a standalone run falls
    # back to the iters-keyed layout the older plots use
    return out_dir if out_dir else f"./plots/run_iters{iters}"


def save_figure(fig, directory, values, name_keys):
    """Write one figure under the name the registry derives for it."""
    os.makedirs(directory, exist_ok=True)
    out = os.path.join(directory, stem(values, name_keys) + ".png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")
    # a sweep stays in one process, so an unclosed figure per grid point piles up
    plt.close(fig)


def control_series(params, traj):
    w, b = params
    states = (
        traj if isinstance(traj, torch.Tensor) else torch.as_tensor(np.asarray(traj))
    )
    return (states.detach().to(torch.float64) @ w.detach() + b.detach()).numpy()


def settings_caption(**settings):
    return "   ".join(f"{key}={value}" for key, value in settings.items())


def add_caption(fig, caption):
    if caption:
        fig.text(0.5, 0.01, caption, ha="center", fontsize=8, color="gray")


def rollout_closed_loop(params, state0=(0, 1, 1.05), steps=300, integrator=euler_step):
    traj = rollout_torch(
        state0,
        lambda s: linear_policy(params, s),
        steps=steps,
        integrator=integrator,
    )
    return traj.detach().numpy(), control_series(params, traj)


def plot_attractor(trajs):
    ax = plt.figure().add_subplot(projection="3d")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")

    for traj in trajs:
        ax.plot(*traj.T, lw=0.5)


def plot_x_vs_t(trajs):
    ax = plt.figure().add_subplot()
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    times = np.arange(len(trajs[0].T[0])) * DT * LYAPUNOV_EXP

    for i, traj in enumerate(trajs):
        xs = traj.T[0]
        ax.plot(times, xs, linestyle="-", linewidth=0.5, label=f"p{i + 1}")

    ax.set_xlabel("Lyapunov times (t / τ)")
    ax.set_ylabel("x")
    ax.legend()


def plot_soft_step_vs_t(trajs, softness=SOFTNESS):
    ax = plt.figure().add_subplot()
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    times = np.arange(len(trajs[0].T[0])) * DT * LYAPUNOV_EXP

    for i, traj in enumerate(trajs):
        ax.plot(
            times,
            soft_step(traj.T[0], softness),
            linestyle="-",
            linewidth=0.5,
            label=f"p{i + 1}",
        )

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Lyapunov times (t / τ)")
    ax.set_ylabel("φ")
    ax.set_title(f"Soft lobe indicator (ε = {softness})")
    ax.legend()


def plot_derivative_separation(separation):
    ax = plt.figure().add_subplot()
    ax.semilogy(
        np.arange(len(separation)) * DT,
        separation,
        label="Exponential Growth",
        color="blue",
        linewidth=0.5,
    )

    ax.set_xlabel("time")
    ax.set_ylabel("separation ‖∇1 − ∇2‖")
    ax.set_title("Gradient divergence")


def plot_lorenz_overview():
    traj_u0, derivs_u0 = rollout_numpy([0, 1, 1.05])
    traj_u1, derivs_u1 = rollout_numpy([0, 1, 1.05], u=1)

    trajs = [traj_u0, traj_u1]
    separation = np.linalg.norm(derivs_u0 - derivs_u1, axis=1)

    plot_attractor(trajs)
    plot_x_vs_t(trajs)
    plot_soft_step_vs_t(trajs)
    plot_derivative_separation(separation)
    plt.show()


def plot_state_and_control(
    traj,
    u,
    penalize_effort=False,
    caption=None,
    axes=None,
):
    # COLORS = ("green", "orange", "black")
    times = np.arange(len(traj)) * DT * LYAPUNOV_EXP

    own_figure = axes is None

    if own_figure:
        _, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))

    ax_traj, ax_u = axes

    ax_traj.axhline(0, color="black", linestyle="--", linewidth=1)
    for i, label in enumerate("xyz"):
        ax_traj.plot(times, traj[:, i], linewidth=0.6, label=label)
    ax_traj.set_ylabel("state")

    title = "Controlled Lorenz trajectory"

    if penalize_effort:
        title += " (regularized)"

    ax_traj.set_title(title)
    ax_traj.legend(loc="upper right")

    ax_u.axhline(0, color="black", linestyle="--", linewidth=1)
    ax_u.plot(times, u, color="crimson", linewidth=0.6)
    ax_u.set_ylabel("control u = w·s + b")
    ax_u.set_xlabel("Lyapunov times (t / τ)")
    ax_u.set_title("Controller action")

    if own_figure:
        fig = ax_traj.figure
        fig.tight_layout(rect=(0, 0.03, 1, 1) if caption else None)
        add_caption(fig, caption)


def plot_attractor_3d(
    traj,
    u,
    penalize_effort=False,
    caption=None,
    ax=None,
):
    own_figure = ax is None

    if own_figure:
        ax = plt.figure(figsize=(11, 7)).add_subplot(projection="3d")

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    title = "Controlled trajectory"

    if penalize_effort:
        title += " (regularized)"
    ax.set_title(title)

    segments = np.stack([traj[:-1], traj[1:]], axis=1)
    lc = Line3DCollection(segments, cmap="coolwarm", linewidths=0.8)
    lc.set_array(u[:-1])
    ax.add_collection(lc)

    ax.auto_scale_xyz(traj[:, 0], traj[:, 1], traj[:, 2])

    ax.figure.colorbar(lc, ax=ax, shrink=0.6, pad=0.11, label="control u")

    if own_figure:
        add_caption(ax.figure, caption)


def plot_state_sensitivity_vs_horizon(state0, u_tensor, max_horizon=1.0, coord=0):
    horizons = np.linspace(0.1, max_horizon, 40)
    grads = [
        final_state_sensitivity(state0, u_tensor, horizon=horizon, coord=coord)[1]
        for horizon in horizons
    ]

    ax = plt.figure().add_subplot()
    ax.semilogy(horizons, np.abs(grads), color="red", linewidth=0.5)
    ax.plot(
        horizons,
        np.exp(np.asarray(horizons)),
        linestyle="--",
        color="gray",
        linewidth=0.5,
        label="e^(t/τ)",
    )

    ax.set_xlabel("Lyapunov times (t / τ)")
    ax.set_ylabel("|∂x(t) / ∂u|")
    ax.set_title("Gradient growth vs. horizon")
    ax.legend()
    plt.show()


def plot_loss_gradient_vs_horizon(
    state0=(0, 1, 1.05), max_horizon=8.0, n=50, save=False, integrator=euler_step
):
    horizons = np.linspace(0.1, max_horizon, n)
    norms = []
    for horizon in horizons:
        params = init_policy_params()
        steps = round(horizon / (LYAPUNOV_EXP * DT))
        traj = rollout_torch(
            state0,
            lambda s: linear_policy(params, s),
            steps=steps,
            integrator=integrator,
        )
        grads = torch.autograd.grad(task_loss(traj), params)
        norms.append(torch.cat([g.reshape(-1) for g in grads]).norm().item())

    ax = plt.figure().add_subplot()
    ax.semilogy(horizons, norms, color="red", linewidth=0.8)
    ax.set_xlabel("Lyapunov times (t / τ)")
    ax.set_ylabel("Loss-gradient")
    ax.set_title("Loss-gradient blowup vs. training window")

    ax.figure.tight_layout(rect=(0, 0.06, 1, 1))
    add_caption(
        ax.figure,
        settings_caption(
            initial_condition=f"({','.join(str(v) for v in state0)})",
            max_horizon=max_horizon,
            n=n,
            dt=DT,
            rk4="on" if integrator is rk4_step else "off",
        ),
    )

    if save:
        # int(max_horizon) used to collide 8.0 with 8.4 and say nothing else
        scheme = "rk4" if integrator is rk4_step else "euler"
        name = f"maxh{decimal_tag(max_horizon)}_n{n}_int-{scheme}"
        path = "./plots/gradient_growth"
        os.makedirs(path, exist_ok=True)
        out = os.path.join(path, name + ".png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        plt.show()


def plot_loss_curve(
    history,
    state0=[0, 1, 1.05],
    lr=0.05,
    train_horizon=1.0,
    plot_horizon=100,
    iters=600,
    penalize_effort=False,
    effort_weight=DEFAULT_EFFORT_WEIGHT,
    integrator=euler_step,
    save=False,
    out_dir=None,
    name_keys=None,
):
    task, penalty, total = np.asarray(history).T
    iterations = np.arange(len(total))

    ax = plt.figure(figsize=(9, 6)).add_subplot()
    ax.plot(iterations, total, color="crimson", linewidth=1.0, label="total loss")

    # if penalize_effort:
    #     ax.plot(iterations, task, color="tab:blue", linewidth=0.8, label="task")
    #     ax.plot(
    #         iterations, penalty, color="gray", linewidth=0.8, label=f"λ·effort"
    #     )

    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title("Training loss vs. iteration")
    ax.legend(loc="upper right")

    print(f"loss: {total[0]:.4f} -> {total[-1]:.4f}   min {total.min():.4f}")

    values = run_values(
        initial_condition=tuple(state0),
        learning_rate=lr,
        train_horizon=train_horizon,
        plot_horizon=plot_horizon,
        iters=iters,
        effort_weight=effort_weight,
        penalize_effort=penalize_effort,
        rk4=integrator is rk4_step,
    )
    caption = settings_caption(**shown(values), dt=DT)

    ax.figure.tight_layout(rect=(0, 0.04, 1, 1))
    add_caption(ax.figure, caption)

    if save:
        # same stem as the attractor plots, so the two directories line up
        save_figure(
            ax.figure,
            os.path.join(run_output_dir(iters, out_dir), "loss_v_iteration"),
            values,
            name_keys or SWEEPABLE,
        )
    else:
        plt.show()


def sweep_axis_order(path):
    """The order the sweep varied its axes in, from the manifest beside the csv.

    Without it the varying columns come back in registry order, which has
    nothing to do with which axis the reader thinks of as x.
    """
    manifest = os.path.join(os.path.dirname(path) or ".", "manifest.json")

    try:
        with open(manifest) as f:
            return list(json.load(f).get("axes", {}))
    except (OSError, ValueError):
        return []


def read_sweep_csv(path):
    """Rows with a success value, plus which registry parameters actually vary."""
    with open(path, newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["success"]]

    if not rows:
        raise SystemExit(f"no success values in {path}")

    columns = [name for name in rows[0] if name in BY_NAME]
    # a column the grid held fixed belongs in the caption, not on an axis
    varying = [
        name
        for name in columns
        if name in SWEEPABLE and len({row[name] for row in rows}) > 1
    ]
    order = sweep_axis_order(path)
    varying.sort(key=lambda name: order.index(name) if name in order else len(order))

    return rows, columns, varying


def plot_success(path, x_key=None, panel_key=None, save=False):
    """Plot the sweep's success metric against whichever parameters it varied.

    x_key goes on the x axis and panel_key gets one panel per value, shaded by
    that value. Both default to the axes the csv shows varying, so a sweep over
    any pair of parameters plots without arguments. The panels share x and y
    limits, so a panel is readable alone and still comparable to its neighbours.
    """
    rows, columns, varying = read_sweep_csv(path)

    x_key = resolve(x_key).name if x_key else (varying[0] if varying else None)

    if x_key is None:
        raise SystemExit(f"nothing varies in {path}; there is no axis to plot")

    if panel_key:
        panel_key = resolve(panel_key).name
    else:
        panel_key = next((name for name in varying if name != x_key), None)

    # a third axis can't be drawn here, and averaging it into the same line
    # silently would be worse than saying so
    hidden = [name for name in varying if name not in (x_key, panel_key)]

    if hidden:
        print(
            f"warning: {', '.join(hidden)} also varies and is not on this plot; "
            f"each point shown is one arbitrary run. Pass -x/-p to choose the "
            f"axes, or filter the csv first."
        )

    x_param = BY_NAME[x_key]

    def numeric(name, row):
        return float(BY_NAME[name].parse(row[name]))

    xs_all = sorted({numeric(x_key, row) for row in rows})
    # a single-parameter sweep is just the one panel
    panel_values = (
        sorted({numeric(panel_key, row) for row in rows}) if panel_key else [None]
    )
    success = {
        (numeric(x_key, row), numeric(panel_key, row) if panel_key else None): float(
            row["success"]
        )
        for row in rows
    }

    def series_for(panel_value):
        xs = [x for x in xs_all if (x, panel_value) in success]

        return xs, [success[(x, panel_value)] for x in xs]

    def panel_label(value):
        return BY_NAME[panel_key].show(BY_NAME[panel_key].parse(value))

    # one panel has nothing to shade against, and Normalize(v, v) divides by zero
    norm = (
        plt.Normalize(min(panel_values), max(panel_values))
        if panel_key and len(panel_values) > 1
        else None
    )
    cmap = plt.colormaps["viridis"]

    n_cols = min(4, len(panel_values))
    n_rows = -(-len(panel_values) // n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.6 * n_cols, 2.9 * n_rows),
        sharex=True,
        sharey=True,
    )
    panels = np.atleast_1d(axes).ravel()

    for ax, panel_value in zip(panels, panel_values):
        ax.plot(
            *series_for(panel_value),
            color=cmap(norm(panel_value)) if norm else cmap(0.5),
            linewidth=1.8,
            marker="o",
            markersize=4,
        )

        if panel_key:
            ax.set_title(f"{panel_key} = {panel_label(panel_value)}", fontsize=10)

        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.25, linewidth=0.6)

    for ax in panels[len(panel_values) :]:
        ax.set_visible(False)

    # sharex hides tick labels everywhere but the bottom row, so the panels in
    # a column that the grid leaves short would otherwise lose their x axis
    for column in range(n_cols):
        column_panels = [i for i in range(column, len(panel_values), n_cols)]
        if column_panels:
            panels[column_panels[-1]].tick_params(labelbottom=True)

    fig.supxlabel(x_param.label, y=0.03)
    fig.supylabel("success (fraction of time x > 0)")

    title = f"Success vs. {x_param.label}"

    if panel_key:
        title += f", by {BY_NAME[panel_key].label}"

    fig.suptitle(title)

    for panel_value in panel_values:
        xs, ys = series_for(panel_value)
        where = "" if panel_key is None else f"{panel_key}={panel_label(panel_value)}: "
        print(f"{where}best success {max(ys):.4f} at {x_key}={xs[ys.index(max(ys))]}")

    # everything the grid held fixed, read back through the registry so a
    # boolean column reads 'on' rather than '1'
    settings = rows[0]
    constant = {
        name: BY_NAME[name].show(BY_NAME[name].parse(settings[name]))
        for name in columns
        if name not in varying
    }

    fig.tight_layout(rect=(0, 0.055, 1, 1))
    add_caption(fig, settings_caption(**constant, dt=DT, runs=len(rows)))

    if save:
        out = os.path.join(os.path.dirname(path), f"success_v_{x_param.key}.png")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        plt.show()


def plot_run_summary(
    params,
    state0=[0, 1, 1.05],
    lr=0.05,
    train_horizon=1.0,
    plot_horizon=25,
    iters=600,
    penalize_effort=False,
    save=False,
    effort_weight=DEFAULT_EFFORT_WEIGHT,
    integrator=euler_step,
    traj=None,
    u=None,
    out_dir=None,
    name_keys=None,
):
    # a caller that batched the rollout already holds the trajectory, and
    # re-integrating it one lambda at a time would undo the point of batching
    if traj is None:
        steps = round(plot_horizon / (LYAPUNOV_EXP * DT))
        traj, u = rollout_closed_loop(
            params, state0=state0, steps=steps, integrator=integrator
        )

    success = success_fraction(traj)
    # sweep.py scrapes this line, so keep the prefix stable
    print(f"success (fraction of time x > 0): {success:.6f}")

    values = run_values(
        initial_condition=tuple(state0),
        learning_rate=lr,
        train_horizon=train_horizon,
        plot_horizon=plot_horizon,
        iters=iters,
        effort_weight=effort_weight,
        penalize_effort=penalize_effort,
        rk4=integrator is rk4_step,
    )
    caption = settings_caption(**shown(values), dt=DT, success=f"{success:.3f}")

    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=(1.1, 1))
    ax_attractor = fig.add_subplot(gs[:, 0], projection="3d")
    ax_traj = fig.add_subplot(gs[0, 1])
    ax_u = fig.add_subplot(gs[1, 1], sharex=ax_traj)

    plot_state_and_control(
        traj, u, penalize_effort=penalize_effort, axes=(ax_traj, ax_u)
    )
    plot_attractor_3d(traj, u, penalize_effort=penalize_effort, ax=ax_attractor)

    fig.tight_layout(rect=(0, 0.03, 1, 1) if caption else None)
    add_caption(fig, caption)

    if save:
        save_figure(
            fig,
            os.path.join(run_output_dir(iters, out_dir), "attractor"),
            values,
            name_keys or SWEEPABLE,
        )
    else:
        plt.show()
