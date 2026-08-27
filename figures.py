"""Every figure the project draws, for both the free and the controlled system.

Nothing here parses arguments; the run_* modules compose these into a run.
"""

import csv
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
from training import (
    DEFAULT_EFFORT_WEIGHT,
    SOFTNESS,
    final_state_sensitivity,
    init_policy_params,
    linear_policy,
    soft_step,
    success_fraction,
    task_loss,
)


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
        plt.savefig(
            f"./plots/gradient_growth/{int(max_horizon)}.png",
            dpi=150,
            bbox_inches="tight",
        )
        print(f"Saved to ./plots/gradient_growth/{int(max_horizon)}.png")
    else:
        plt.show()


def plot_loss_curve(
    history,
    state0=[0, 1, 1.05],
    lr=0.05,
    train_horizon=1.0,
    iters=600,
    penalize_effort=False,
    effort_weight=DEFAULT_EFFORT_WEIGHT,
    integrator=euler_step,
    save=False,
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

    caption = settings_caption(
        initial_condition=f"({','.join(str(v) for v in state0)})",
        learning_rate=lr,
        train_horizon=train_horizon,
        iters=iters,
        effort_weight=effort_weight,
        dt=DT,
        penalize_effort="on" if penalize_effort else "off",
        rk4="on" if integrator is rk4_step else "off",
    )

    ax.figure.tight_layout(rect=(0, 0.04, 1, 1))
    add_caption(ax.figure, caption)

    if save:
        path = f"./plots/loss_sweep_{iters}/loss_v_iteration/"
        # same stem as the attractor plots so the two directories line up
        filename = f"lambda{effort_weight}_{train_horizon}"
        filename += "_regularized" if penalize_effort else "_unregularized"

        if integrator is rk4_step:
            filename += "_rk4"

        filename += "_loss"
        os.makedirs(path, exist_ok=True)
        plt.savefig(path + filename + ".png", dpi=150, bbox_inches="tight")
        print(f"Saved to {path + filename}.png")
        plt.close(ax.figure)
    else:
        plt.show()


def plot_success_vs_lambda(path, save=False):
    """Plot the sweep's success metric against the regularizer strength λ.

    One panel per training window, each shaded by window length. The panels
    share x and y limits and carry the rest of the sweep behind them in grey,
    so a panel is readable on its own and still comparable to its neighbours.
    """
    with open(path, newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["success"]]

    if not rows:
        raise SystemExit(f"no success values in {path}")

    effort_weights = sorted({float(row["effort_weight"]) for row in rows})
    train_horizons = sorted({float(row["train_horizon"]) for row in rows})
    success = {
        (float(row["effort_weight"]), float(row["train_horizon"])): float(
            row["success"]
        )
        for row in rows
    }

    def series_for(train_horizon):
        xs = [weight for weight in effort_weights if (weight, train_horizon) in success]

        return xs, [success[(weight, train_horizon)] for weight in xs]

    norm = plt.Normalize(min(train_horizons), max(train_horizons))
    cmap = plt.colormaps["viridis"]

    n_cols = min(4, len(train_horizons))
    n_rows = -(-len(train_horizons) // n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.6 * n_cols, 2.9 * n_rows),
        sharex=True,
        sharey=True,
    )
    panels = np.atleast_1d(axes).ravel()

    for ax, train_horizon in zip(panels, train_horizons):
        # the other windows in grey give each panel back the context it loses

        ax.plot(
            *series_for(train_horizon),
            color=cmap(norm(train_horizon)),
            linewidth=1.8,
            marker="o",
            markersize=4,
        )

        ax.set_title(f"train_horizon = {train_horizon}", fontsize=10)
        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.25, linewidth=0.6)

    for ax in panels[len(train_horizons) :]:
        ax.set_visible(False)

    # sharex hides tick labels everywhere but the bottom row, so the panels in
    # a column that the grid leaves short would otherwise lose their x axis
    for column in range(n_cols):
        column_panels = [i for i in range(column, len(train_horizons), n_cols)]
        if column_panels:
            panels[column_panels[-1]].tick_params(labelbottom=True)

    fig.supxlabel("λ (control-effort penalty)", y=0.03)
    fig.supylabel("success (fraction of time x > 0)")
    fig.suptitle("Success vs. regularizer strength, by training window")

    for train_horizon in train_horizons:
        xs, ys = series_for(train_horizon)
        print(
            f"train_horizon={train_horizon}: best success {max(ys):.4f} "
            f"at effort_weight={xs[ys.index(max(ys))]}"
        )

    settings = rows[0]
    caption = settings_caption(
        learning_rate=settings["learning_rate"],
        iters=settings["iters"],
        plot_horizon=settings["plot_horizon"],
        dt=DT,
        penalize_effort="on" if settings["penalize_effort"] == "1" else "off",
        rk4="on" if settings["rk4"] == "1" else "off",
        runs=len(rows),
    )

    fig.tight_layout(rect=(0, 0.055, 1, 1))
    add_caption(fig, caption)

    if save:
        out = os.path.join(os.path.dirname(path), "success_v_lambda.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
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

    caption = settings_caption(
        initial_condition=f"({','.join(str(v) for v in state0)})",
        learning_rate=lr,
        train_horizon=train_horizon,
        plot_horizon=plot_horizon,
        iters=iters,
        effort_weight=effort_weight,
        dt=DT,
        penalize_effort="on" if penalize_effort else "off",
        rk4="on" if integrator is rk4_step else "off",
        success=f"{success:.3f}",
    )

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
        path = f"./plots/loss_sweep_{iters}/attractor/"
        filename = f"lambda{effort_weight}_{train_horizon}"

        if penalize_effort:
            filename += "_regularized"
        else:
            filename += "_unregularized"

        if integrator == rk4_step:
            filename += "_rk4"

        os.makedirs(path, exist_ok=True)
        plt.savefig(path + filename + ".png", dpi=150, bbox_inches="tight")
        print(f"Saved to {path + filename}.png")
        # a sweep stays in one process, so an unclosed figure per grid point piles up
        plt.close(fig)
    else:
        plt.show()
