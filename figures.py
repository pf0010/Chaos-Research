"""Every figure the project draws, for both the free and the controlled system.

Nothing here parses arguments; the run_* modules compose these into a run.
"""

import csv
import itertools
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from lorenz import (
    DT,
    LYAPUNOV_EXP,
    rk4_step,
    rollout_numpy,
    rollout_torch,
)
from params import (
    BY_NAME,
    DEFAULT_EFFORT_WEIGHT,
    SWEEPABLE,
    decimal_tag,
    run_values,
    shown,
    stem,
)
from training import (
    SOFTNESS,
    effort_penalty,
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


SETTINGS_GAP = "   "


def settings_caption(**settings):
    return SETTINGS_GAP.join(f"{key}={value}" for key, value in settings.items())


def add_caption(fig, caption, y=0.01):
    # y is a figure fraction, so a caller whose figure height varies with its
    # panel count has to scale it to keep the gap the same in inches. The text
    # is anchored by its bottom, so a caption that wrapped grows up into the
    # strip reserved for it rather than down off the canvas
    if caption:
        fig.text(0.5, y, caption, ha="center", va="bottom", fontsize=8, color="gray")


def add_band_legend(fig, y):
    """The key to the ensemble bands, once per figure rather than once per panel.

    Drawn in grey: the swatches say what a band means, not which series it
    belongs to -- each panel keeps its own colour, and repeating one of them
    here would read as picking a favourite.
    """
    handles = [
        Patch(facecolor="0.30", alpha=alpha, linewidth=0, label=f"{low}–{high}%")
        for low, high, alpha in BANDS
    ]
    handles.append(
        Line2D(
            [],
            [],
            color="0.30",
            linewidth=1.8,
            marker="o",
            markersize=4,
            label="median",
        )
    )

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=len(handles),
        frameon=False,
        fontsize=8,
        handlelength=1.6,
        columnspacing=1.6,
    )


def wrap_caption(caption, width):
    """Break a settings caption over as many lines as `width` inches will hold.

    A caption wider than its figure is not merely ugly: savefig's tight bounding
    box grows to contain it, so the panels end up in the left corner of a canvas
    sized by the text. 8pt runs about 18 characters to the inch, and the caption
    is a run of key=value pairs, so it breaks cleanly between them.
    """
    limit = max(40, int(width * 18))
    lines = []

    for part in caption.split(SETTINGS_GAP):
        if lines and len(lines[-1]) + len(SETTINGS_GAP) + len(part) <= limit:
            lines[-1] += SETTINGS_GAP + part
        else:
            lines.append(part)

    return "\n".join(lines)


def rollout_closed_loop(params, state0=(0, 1, 1.05), steps=300, integrator=rk4_step):
    # the trajectory is only ever plotted, and taping a rollout this long costs
    # about as much again as integrating it
    with torch.no_grad():
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
    penalize_effort=True,
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
    penalize_effort=True,
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
    state0=(0, 1, 1.05), max_horizon=8.0, n=50, save=False, integrator=rk4_step
):
    horizons = np.linspace(0.1, max_horizon, n)
    norms = []
    # one draw for every horizon, so the curve moves with the window alone
    params = init_policy_params()
    for horizon in horizons:
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
    penalize_effort=True,
    effort_weight=DEFAULT_EFFORT_WEIGHT,
    integrator=rk4_step,
    save=False,
    out_dir=None,
    name_keys=None,
    seed=None,
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

    finish_training_curve(
        ax.figure,
        "loss_v_iteration",
        state0=state0,
        lr=lr,
        train_horizon=train_horizon,
        plot_horizon=plot_horizon,
        iters=iters,
        penalize_effort=penalize_effort,
        effort_weight=effort_weight,
        integrator=integrator,
        save=save,
        out_dir=out_dir,
        name_keys=name_keys,
        seed=seed,
    )


def plot_grad_norm_curve(
    grad_norm,
    state0=[0, 1, 1.05],
    lr=0.05,
    train_horizon=1.0,
    plot_horizon=100,
    iters=600,
    penalize_effort=True,
    effort_weight=DEFAULT_EFFORT_WEIGHT,
    integrator=rk4_step,
    save=False,
    out_dir=None,
    name_keys=None,
    seed=None,
):
    task, penalty, total = np.asarray(grad_norm).T
    iterations = np.arange(len(total))

    ax = plt.figure(figsize=(9, 6)).add_subplot()
    ax.plot(iterations, total, color="crimson", linewidth=1.0, label="total")
    ax.plot(iterations, task, color="tab:blue", linewidth=0.8, label="task")

    # an unpenalized run never differentiates the penalty, so there is no line
    if penalize_effort:
        ax.plot(iterations, penalty, color="gray", linewidth=0.8, label="λ·effort")

    # through a chaotic window the norms span orders of magnitude
    ax.set_yscale("log")

    ax.set_xlabel("iteration")
    ax.set_ylabel("‖∇‖₂  over (w, b)")
    ax.set_title("Gradient norms vs. iteration")
    ax.legend(loc="upper right")

    print(
        f"grad norm: {total[0]:.4g} -> {total[-1]:.4g}   max {total.max():.4g}   "
        f"final task {task[-1]:.4g}   λ·effort {penalty[-1]:.4g}"
    )

    finish_training_curve(
        ax.figure,
        "grad_norm_v_iteration",
        state0=state0,
        lr=lr,
        train_horizon=train_horizon,
        plot_horizon=plot_horizon,
        iters=iters,
        penalize_effort=penalize_effort,
        effort_weight=effort_weight,
        integrator=integrator,
        save=save,
        out_dir=out_dir,
        name_keys=name_keys,
        seed=seed,
    )


def plot_state_gradient_through_window(
    param_history,
    state0=[0, 1, 1.05],
    lr=0.05,
    train_horizon=1.0,
    plot_horizon=100,
    iters=600,
    penalize_effort=True,
    effort_weight=DEFAULT_EFFORT_WEIGHT,
    integrator=rk4_step,
    save=False,
    out_dir=None,
    name_keys=None,
    seed=None,
    max_rows=150,
    starts=None,
):
    """‖∂L/∂s_t‖ at every step of the training window, across training.

    This is the gradient as backpropagation carries it back through the
    rollout, so chaos shows up as growth from the end of the window towards
    its start. Each recorded iteration's (w, b) is a lane of one float64
    rollout -- the lanes are independent, so one backward pass hands every
    state its own lane's gradient. `max_rows` strides the iterations to keep
    the saved activations bounded.

    Under receding-horizon training each window starts somewhere new, so
    `starts` -- one per window, iters rows each -- says where each row's
    rollout begins; without it every row begins at `state0`. A shortened
    last window is replayed over the full window, which only this
    diagnostic sees.
    """
    seen = np.asarray(param_history, dtype=float)
    stride = max(1, math.ceil(len(seen) / max_rows))
    rows = np.arange(0, len(seen), stride)

    w = torch.as_tensor(seen[rows, :3])
    b = torch.as_tensor(seen[rows, 3])
    steps = round(train_horizon / (LYAPUNOV_EXP * DT))

    # the same objective training differentiated: every window's penalty is
    # pathwise over that window, so it reaches the states like the task does
    pathwise = penalize_effort

    row_starts = (
        np.asarray(starts, dtype=float)[rows // iters]
        if starts is not None
        else np.broadcast_to(np.asarray(state0, dtype=float), (len(rows), 3))
    )
    state = torch.as_tensor(row_starts).clone().requires_grad_()
    states = [state]

    for _ in range(steps):
        state, _ = integrator(state, lambda s: linear_policy((w, b), s))
        state.retain_grad()
        states.append(state)

    traj = torch.stack(states)
    loss = task_loss(traj)

    if pathwise:
        loss = loss + effort_weight * effort_penalty(traj, (w, b))

    loss.sum().backward()

    # (steps+1, rows): the full gradient reaching each state, every later use
    # of it included, not just its own term in the loss
    adjoint = torch.stack([s.grad for s in states]).norm(dim=-1).numpy()
    times = np.arange(steps + 1) * DT * LYAPUNOV_EXP

    fig, (heat, lines) = plt.subplots(1, 2, figsize=(15, 6))

    positive = adjoint[adjoint > 0]
    color_norm = LogNorm(vmin=max(positive.min(), positive.max() * 1e-8), vmax=positive.max())
    mesh = heat.pcolormesh(
        times, rows, adjoint.T, norm=color_norm, cmap="viridis", shading="nearest"
    )
    fig.colorbar(mesh, ax=heat, label="‖∂L/∂s_t‖₂")

    heat.set_xlabel("time in training window (Lyapunov times)")
    heat.set_ylabel("iteration")
    heat.set_title("Gradient w.r.t. the state, across training")

    picks = np.unique(np.linspace(0, len(rows) - 1, 5).round().astype(int))
    cmap = plt.colormaps["viridis"]

    for j, i in enumerate(picks):
        lines.plot(
            times,
            adjoint[:, i],
            color=cmap(j / max(len(picks) - 1, 1)),
            linewidth=1.0,
            label=f"iteration {rows[i]}",
        )

    # what an uncontrolled perturbation would do: one e-fold per Lyapunov
    # time, run backwards from the end of the window. Only its slope means
    # anything -- the gradient also grows backwards because every later
    # step's loss term piles onto it
    lines.plot(
        times,
        adjoint[-1, 0] * np.exp(times[-1] - times),
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label="slope of e^(T−t)  (Lyapunov growth)",
    )

    lines.set_yscale("log")
    lines.set_ylim(color_norm.vmin, color_norm.vmax * 10)
    lines.set_xlabel("time in training window (Lyapunov times)")
    lines.set_ylabel("‖∂L/∂s_t‖₂")
    lines.set_title("Selected iterations")
    lines.legend(loc="upper right")

    for i in (0, len(rows) - 1):
        print(
            f"iteration {rows[i]}: ‖∂L/∂s_0‖ {adjoint[0, i]:.3g}   "
            f"‖∂L/∂s_T‖ {adjoint[-1, i]:.3g}"
        )

    finish_training_curve(
        fig,
        "state_gradient_v_time",
        state0=state0,
        lr=lr,
        train_horizon=train_horizon,
        plot_horizon=plot_horizon,
        iters=iters,
        penalize_effort=penalize_effort,
        effort_weight=effort_weight,
        integrator=integrator,
        save=save,
        out_dir=out_dir,
        name_keys=name_keys,
        seed=seed,
    )


def finish_training_curve(
    fig,
    subdir,
    state0,
    lr,
    train_horizon,
    plot_horizon,
    iters,
    penalize_effort,
    effort_weight,
    integrator,
    save,
    out_dir,
    name_keys,
    seed=None,
):
    """Caption a per-iteration figure, then save it under `subdir` or show it."""
    values = run_values(
        initial_condition=tuple(state0),
        seed=seed,
        learning_rate=lr,
        train_horizon=train_horizon,
        plot_horizon=plot_horizon,
        iters=iters,
        effort_weight=effort_weight,
        penalize_effort=penalize_effort,
        rk4=integrator is rk4_step,
    )
    caption = settings_caption(**shown(values), dt=DT)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    add_caption(fig, caption)

    if save:
        # same stem as the attractor plots, so the directories line up
        save_figure(
            fig,
            os.path.join(run_output_dir(iters, out_dir), subdir),
            values,
            name_keys or SWEEPABLE,
        )
    else:
        plt.show()


# the success figure's axes are fixed: a sweep is only ever read as success
# against the training window, so the two of them are not the caller's choice
X_KEY = "train_horizon"
# ...and these are ensemble axes rather than hyperparameters. Runs that differ
# only in where they started -- a start given outright or one drawn from a
# seed -- are averaged into one line, not split apart. A seeded run records
# the start it drew, so its initial_condition varies with the seed and has to
# be left out of the organizing too
ENSEMBLE_KEYS = ("initial_condition", "seed")
# four columns by six rows is about as much as a page holds and stays legible
MAX_PANELS = 24

# What an ensemble of starts is drawn as: nested percentile bands, palest
# outermost, under a median line. Two bands rather than one because the outer
# envelope alone says only how bad the worst start was, while the inner one
# says whether the bulk of them agreed -- a wide outer band around a tight
# inner one is a couple of unlucky starts, a wide inner one is a real spread.
# The median rather than the mean, so a single ruined start moves the band it
# belongs in instead of the centre.
BANDS = ((5, 95, 0.15), (25, 75, 0.32))


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
    # a column the grid held fixed belongs in the caption, not on an axis.
    # Compared as parsed values rather than as text: '0|1|1.05' and
    # '0.0|1.0|1.05' are the same start, and two csvs concatenated for an
    # ensemble are exactly where the two spellings meet
    varying = [
        name
        for name in columns
        if len({BY_NAME[name].parse(row[name]) for row in rows}) > 1
    ]
    order = sweep_axis_order(path)
    varying.sort(key=lambda name: order.index(name) if name in order else len(order))

    # a csv written by a version of params.py this one no longer has -- a
    # branch, a stash, an older checkout -- carries columns the registry can't
    # place. They can't organize a figure, so they get averaged into one, and
    # that is worth saying out loud rather than leaving to be noticed
    stranded = [
        name
        for name in rows[0]
        if name not in BY_NAME
        and name != "success"
        and len({row[name] for row in rows}) > 1
    ]

    if stranded:
        print(
            f"warning: {', '.join(stranded)} varies in {path} but is not in "
            f"params.py, so it cannot organize the figure; runs differing only "
            f"in it are averaged together"
        )

    return rows, columns, varying


def plot_success(path, save=False):
    """Success against the training window, one panel per hyperparameter combination.

    The axes never move: y is the success metric and x is the training window,
    so two sweeps can be laid side by side and read the same way. Everything
    else the grid varied organizes the figure instead -- one panel per
    combination of them, sharing x and y limits so a panel is readable alone
    and still comparable to its neighbours. A grid too big for one page splits
    over its outermost parameters, a file per value.

    initial_condition and seed are the parameters that do not organize
    anything. Runs that differ only in where they started are the same
    experiment repeated, so they collapse into one line: the median, over
    bands spanning the ensemble.
    """
    rows, columns, varying = read_sweep_csv(path)

    if X_KEY not in varying:
        raise SystemExit(
            f"{X_KEY} does not vary in {path}; this figure is success against "
            f"the training window, so there is nothing to put on its x axis"
        )

    organizing = [
        name for name in varying if name != X_KEY and name not in ENSEMBLE_KEYS
    ]

    def numeric(name, row):
        return float(BY_NAME[name].parse(row[name]))

    def label(name, value):
        param = BY_NAME[name]

        return f"{param.key}={param.show(param.parse(value))}"

    xs_all = sorted({numeric(X_KEY, row) for row in rows})
    values_of = {
        name: sorted({numeric(name, row) for row in rows}) for name in organizing
    }
    # a start is named by whichever of the two the csv carries
    ics = {
        tuple(BY_NAME[name].parse(row.get(name, "")) for name in ENSEMBLE_KEYS)
        for row in rows
    }

    # (the organizing parameters, x) -> every run there, which is one run per
    # initial condition. Anything the csv does not distinguish lands in the
    # same bucket, so a repeated grid point is averaged rather than overwritten
    ensembles = {}

    for row in rows:
        key = (tuple(numeric(name, row) for name in organizing), numeric(X_KEY, row))
        ensembles.setdefault(key, []).append(float(row["success"]))

    # the outermost parameters page the figure until what is left fits, so the
    # panels on a page stay big enough to read. The last one is never paged
    # away: a file per panel is a worse way to read a long axis than a tall
    # grid is, and the axis it sweeps is the point of the figure
    panel_keys = list(organizing)
    page_keys = []

    while (
        len(panel_keys) > 1
        and math.prod(len(values_of[k]) for k in panel_keys) > MAX_PANELS
    ):
        page_keys.append(panel_keys.pop(0))

    pages = list(itertools.product(*(values_of[name] for name in page_keys)))
    panels = list(itertools.product(*(values_of[name] for name in panel_keys)))

    # panels are shaded by the first parameter that still varies within a page;
    # one panel has nothing to shade against, and Normalize(v, v) divides by zero
    shade_key = panel_keys[0] if panel_keys else None
    norm = (
        plt.Normalize(min(values_of[shade_key]), max(values_of[shade_key]))
        if shade_key and len(values_of[shade_key]) > 1
        else None
    )
    cmap = plt.colormaps["viridis"]

    # the innermost parameter varies fastest across the product, so making it
    # the row length puts one of its sweeps on every line of the grid
    innermost = len(values_of[panel_keys[-1]]) if panel_keys else 0
    n_cols = innermost if 2 <= innermost <= 6 else min(4, len(panels))
    n_rows = -(-len(panels) // n_cols)

    # everything the grid held fixed, read back through the registry so a
    # boolean column reads 'on' rather than '1'
    settings = rows[0]
    fixed = {name: BY_NAME[name].parse(settings[name]) for name in columns}
    # an empty cell is a value the run never had -- a given start's seed
    caption = {
        name: BY_NAME[name].show(value)
        for name, value in fixed.items()
        if name not in varying and value is not None
    }
    caption["dt"] = DT
    caption["runs"] = len(rows)

    # the legend says what the bands are, so the caption only has to say how
    # many runs went into them
    spread = len(ics) > 1

    if spread:
        caption["ensemble"] = f"{len(ics)} starts"

    x_param = BY_NAME[X_KEY]
    out_dir = os.path.dirname(path) or "."
    outputs = []

    for page_values in pages:
        page = dict(zip(page_keys, page_values))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            # the title above and the caption below take a fixed bite out of
            # any figure, so they get their own inch rather than a share of
            # the panels'. A lone panel also has a width its labels need
            figsize=(max(3.6 * n_cols, 7.0), 2.9 * n_rows + 1.2),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        grid = axes.ravel()
        best = None

        for ax, combination in zip(grid, panels):
            held = dict(zip(panel_keys, combination))
            key = tuple({**page, **held}[name] for name in organizing)
            xs = [x for x in xs_all if (key, x) in ensembles]
            runs = [ensembles[(key, x)] for x in xs]

            if not xs:
                # the grid is a product, so a combination the sweep skipped
                # leaves a hole rather than shifting its neighbours along
                ax.set_visible(False)
                continue

            centres = [float(np.median(run)) for run in runs]
            color = cmap(norm(held[shade_key])) if norm else cmap(0.5)

            # a single initial condition has no spread, and a zero-height band
            # is just a second line drawn on the first
            if any(len(run) > 1 for run in runs):
                for low, high, alpha in BANDS:
                    ax.fill_between(
                        xs,
                        [float(np.percentile(run, low)) for run in runs],
                        [float(np.percentile(run, high)) for run in runs],
                        color=color,
                        alpha=alpha,
                        linewidth=0,
                    )

            ax.plot(xs, centres, color=color, linewidth=1.8, marker="o", markersize=4)

            if held:
                ax.set_title(
                    "  ".join(label(name, value) for name, value in held.items()),
                    fontsize=9,
                )

            ax.set_ylim(-0.03, 1.05)
            ax.grid(alpha=0.25, linewidth=0.6)

            top = max(centres)

            if best is None or top > best[0]:
                best = (top, xs[centres.index(top)], held)

        for ax in grid[len(panels) :]:
            ax.set_visible(False)

        # sharex hides tick labels everywhere but the bottom row, so the panels
        # in a column that the grid leaves short would otherwise lose their x axis
        for column in range(n_cols):
            visible = [
                i for i in range(column, len(panels), n_cols) if grid[i].get_visible()
            ]

            if visible:
                grid[visible[-1]].tick_params(labelbottom=True)

        # and sharey hides them everywhere but the first column, so a row whose
        # leftmost panel is a combination the sweep never ran loses its y axis
        for start in range(0, len(panels), n_cols):
            visible = [
                i
                for i in range(start, min(start + n_cols, len(panels)))
                if grid[i].get_visible()
            ]

            if visible:
                grid[visible[0]].tick_params(labelleft=True)

        # the figure grows with its panel count, so the strip under the panels
        # is measured in inches and converted, not left at a fixed fraction
        height = fig.get_figheight()
        text = wrap_caption(settings_caption(**caption), fig.get_figwidth())
        # 8pt lines are about 0.14in apart, and the label needs the top of the
        # strip to itself
        below = 0.14 * text.count("\n")
        fig.supxlabel(x_param.label, y=(0.20 + below) / height)
        fig.supylabel("success (fraction of time x > 0)")

        title = f"Success vs. {x_param.label}"

        # the parameter labels are sentences rather than symbols, so three of
        # them on one line would be wider than the panels underneath
        if organizing:
            title += "\nby " + ", ".join(BY_NAME[name].label for name in organizing)
        if page:
            title += "\n" + "   ".join(
                label(name, value) for name, value in page.items()
            )

        # the title and the band key share the strip above the panels, and it
        # is measured in inches for the same reason the one below is: a fixed
        # fraction of a six-row figure is inches of empty space
        lines = title.count("\n") + 1
        above = 0.12 + 0.24 * lines + (0.34 if spread else 0.08)

        # fig.text rather than fig.suptitle: tight_layout reserves space for a
        # suptitle on its own, and doing that on top of the rect below would
        # take the strip twice and leave the panels squashed under a gap
        fig.text(
            0.5,
            1 - 0.12 / height,
            title,
            ha="center",
            va="top",
            fontsize=plt.rcParams["figure.titlesize"],
        )

        if spread:
            add_band_legend(fig, 1 - (0.16 + 0.24 * lines) / height)

        # tight_layout does not know about supylabel, so the left edge is
        # reserved by hand the way the bottom strip is
        fig.tight_layout(
            rect=(
                0.30 / fig.get_figwidth(),
                (0.42 + below) / height,
                1,
                1 - above / height,
            )
        )
        add_caption(fig, text, y=0.06 / height)

        where = "".join(f"{label(name, value)} " for name, value in page.items())

        if best:
            print(
                f"{where}best success {best[0]:.4f} at {X_KEY}={best[1]}"
                + (
                    "  " + "  ".join(label(n, v) for n, v in best[2].items())
                    if best[2]
                    else ""
                )
            )

        if save:
            # one page keeps the flat name the sweep directory has always had;
            # more than one gets a directory of its own, named the way the
            # attractor plots are
            if len(pages) > 1:
                out = os.path.join(
                    out_dir,
                    f"success_v_{x_param.key}",
                    stem(page, page_keys) + ".png",
                )
            else:
                out = os.path.join(out_dir, f"success_v_{x_param.key}.png")

            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            outputs.append(out)

    if not save:
        plt.show()
    elif len(outputs) == 1:
        print(f"Saved to {outputs[0]}")
    else:
        print(f"Saved {len(outputs)} pages to {os.path.dirname(outputs[0])}/")


def plot_run_summary(
    params,
    state0=[0, 1, 1.05],
    lr=0.05,
    train_horizon=1.0,
    plot_horizon=25,
    iters=600,
    penalize_effort=True,
    save=False,
    effort_weight=DEFAULT_EFFORT_WEIGHT,
    integrator=rk4_step,
    traj=None,
    u=None,
    out_dir=None,
    name_keys=None,
    seed=None,
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
        seed=seed,
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
