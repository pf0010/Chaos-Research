from sim import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import argparse
import csv
import os


def controller_actions(points, params):
    w, b = params
    pts = (
        points
        if isinstance(points, torch.Tensor)
        else torch.as_tensor(np.asarray(points))
    )
    return (pts.detach().to(torch.float64) @ w.detach() + b.detach()).numpy()


def settings_caption(**settings):
    return "   ".join(f"{key}={value}" for key, value in settings.items())


def add_caption(fig, caption):
    if caption:
        fig.text(0.5, 0.01, caption, ha="center", fontsize=8, color="gray")


def control_trajectory(params, initial=(0, 1, 1.05), steps=300, integrator=euler_step):
    traj = tensor_data(
        *initial,
        lambda s: control_force(s, params),
        steps=steps,
        integrator=integrator,
    )
    return traj.detach().numpy(), controller_actions(traj, params)


def plot_control(
    pts,
    us,
    regularized=False,
    caption=None,
    axes=None,
):
    # COLORS = ("green", "orange", "black")
    lyapunov_times = np.arange(len(pts)) * DT * LYAPUNOV_EXP

    own_figure = axes is None

    if own_figure:
        _, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))

    ax_traj, ax_u = axes

    ax_traj.axhline(0, color="black", linestyle="--", linewidth=1)
    for i, label in enumerate("xyz"):
        ax_traj.plot(lyapunov_times, pts[:, i], linewidth=0.6, label=label)
    ax_traj.set_ylabel("state")

    title = "Controlled Lorenz trajectory"

    if regularized:
        title += " (regularized)"

    ax_traj.set_title(title)
    ax_traj.legend(loc="upper right")

    ax_u.axhline(0, color="black", linestyle="--", linewidth=1)
    ax_u.plot(lyapunov_times, us, color="crimson", linewidth=0.6)
    ax_u.set_ylabel("control u = w·s + b")
    ax_u.set_xlabel("Lyapunov times (t / τ)")
    ax_u.set_title("Controller action")

    if own_figure:
        fig = ax_traj.figure
        fig.tight_layout(rect=(0, 0.03, 1, 1) if caption else None)
        add_caption(fig, caption)


def plot_control_attractor(
    pts,
    us,
    regularized=False,
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

    if regularized:
        title += " (regularized)"
    ax.set_title(title)

    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = Line3DCollection(segments, cmap="coolwarm", linewidths=0.8)
    lc.set_array(us[:-1])
    ax.add_collection(lc)

    ax.auto_scale_xyz(pts[:, 0], pts[:, 1], pts[:, 2])

    ax.figure.colorbar(lc, ax=ax, shrink=0.6, pad=0.11, label="control u")

    if own_figure:
        add_caption(ax.figure, caption)


def plot_gradient_growth(
    initial_x, initial_y, initial_z, ut, lyapunov_times=1.0, coord=0
):
    horizons = np.linspace(0.1, lyapunov_times, 40)
    grads = [
        position_gradient(
            initial_x, initial_y, initial_z, ut, lyapunov_times=lt, coord=coord
        )[1]
        for lt in horizons
    ]

    fig = plt.figure().add_subplot()
    fig.semilogy(horizons, np.abs(grads), color="red", linewidth=0.5)
    fig.plot(
        horizons,
        np.exp(np.asarray(horizons)),
        linestyle="--",
        color="gray",
        linewidth=0.5,
        label="e^(t/τ)",
    )

    fig.set_xlabel("Lyapunov times (t / τ)")
    fig.set_ylabel("|∂x(t) / ∂u|")
    fig.set_title("Gradient growth vs. horizon")
    fig.legend()
    plt.show()


def plot_gradient_vs_window(
    ic=(0, 1, 1.05), max_lt=8.0, n=50, save=False, integrator=euler_step
):
    windows = np.linspace(0.1, max_lt, n)
    norms = []
    for lt in windows:
        params = create_controller()
        steps = round(lt / (LYAPUNOV_EXP * DT))
        traj = tensor_data(
            *ic,
            lambda s: control_force(s, params),
            steps=steps,
            integrator=integrator,
        )
        g = torch.autograd.grad(calculate_loss(traj), params)
        norms.append(torch.cat([gi.reshape(-1) for gi in g]).norm().item())

    ax = plt.figure().add_subplot()
    ax.semilogy(windows, norms, color="red", linewidth=0.8)
    ax.set_xlabel("Lyapunov times (t / τ)")
    ax.set_ylabel("Loss-gradient")
    ax.set_title("Loss-gradient blowup vs. training window")

    ax.figure.tight_layout(rect=(0, 0.06, 1, 1))
    add_caption(
        ax.figure,
        settings_caption(
            ic=f"({','.join(str(v) for v in ic)})",
            gg=max_lt,
            n=n,
            dt=DT,
            rk4="on" if integrator is rk4_step else "off",
        ),
    )

    if save:
        plt.savefig(
            f"./plots/gradient_growth/{int(max_lt)}.png", dpi=150, bbox_inches="tight"
        )
        print(f"Saved to ./plots/gradient_growth/{int(max_lt)}.png")
    else:
        plt.show()


def plot_loss_curve(
    history,
    ic=[0, 1, 1.05],
    lr=0.05,
    train_lyapunov_times=1.0,
    iters=600,
    regularized=False,
    lam=LAMBDA,
    integrator=euler_step,
    save=False,
):
    task, penalty, total = np.asarray(history).T
    iterations = np.arange(len(total))

    ax = plt.figure(figsize=(9, 6)).add_subplot()
    ax.plot(iterations, total, color="crimson", linewidth=1.0, label="total loss")

    # if regularized:
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
        ic=f"({','.join(str(v) for v in ic)})",
        lr=lr,
        tlt=train_lyapunov_times,
        iters=iters,
        lam=lam,
        dt=DT,
        l2="on" if regularized else "off",
        rk4="on" if integrator is rk4_step else "off",
    )

    ax.figure.tight_layout(rect=(0, 0.04, 1, 1))
    add_caption(ax.figure, caption)

    if save:
        path = f"./plots/loss_sweep_{iters}/loss_v_iteration/"
        # same stem as the attractor plots so the two directories line up
        filename = f"lambda{lam}_{train_lyapunov_times}"
        filename += "_regularized" if regularized else "_unregularized"

        if integrator is rk4_step:
            filename += "_rk4"

        filename += "_loss"
        os.makedirs(path, exist_ok=True)
        plt.savefig(path + filename + ".png", dpi=150, bbox_inches="tight")
        print(f"Saved to {path + filename}.png")
        plt.close(ax.figure)
    else:
        plt.show()


def plot_success_vs_lambda(path="./plots/loss_sweep_1000/success.csv", save=False):
    """Plot the sweep's success metric against the regularizer strength λ.

    One panel per training window, each shaded by window length. The panels
    share x and y limits and carry the rest of the sweep behind them in grey,
    so a panel is readable on its own and still comparable to its neighbours.
    """
    with open(path, newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["success"]]

    if not rows:
        raise SystemExit(f"no success values in {path}")

    lams = sorted({float(row["lam"]) for row in rows})
    tlts = sorted({float(row["tlt"]) for row in rows})
    success = {(float(row["lam"]), float(row["tlt"])): float(row["success"]) for row in rows}

    def line(tlt):
        xs = [lam for lam in lams if (lam, tlt) in success]

        return xs, [success[(lam, tlt)] for lam in xs]

    norm = plt.Normalize(min(tlts), max(tlts))
    cmap = plt.colormaps["viridis"]

    cols = min(4, len(tlts))
    grid_rows = -(-len(tlts) // cols)

    fig, axes = plt.subplots(
        grid_rows,
        cols,
        figsize=(3.6 * cols, 2.9 * grid_rows),
        sharex=True,
        sharey=True,
    )
    flat = np.atleast_1d(axes).ravel()

    for ax, tlt in zip(flat, tlts):
        # the other windows in grey give each panel back the context it loses

        ax.plot(*line(tlt), color=cmap(norm(tlt)), linewidth=1.8, marker="o", markersize=4)

        ax.set_title(f"tlt = {tlt}", fontsize=10)
        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.25, linewidth=0.6)

    for ax in flat[len(tlts) :]:
        ax.set_visible(False)

    # sharex hides tick labels everywhere but the bottom row, so the panels in
    # a column that the grid leaves short would otherwise lose their x axis
    for column in range(cols):
        bottom = [i for i in range(column, len(tlts), cols)]
        if bottom:
            flat[bottom[-1]].tick_params(labelbottom=True)

    fig.supxlabel("λ (control-effort penalty)", y=0.03)
    fig.supylabel("success (fraction of time x > 0)")
    fig.suptitle("Success vs. regularizer strength, by training window")

    for tlt in tlts:
        xs, ys = line(tlt)
        print(f"tlt={tlt}: best success {max(ys):.4f} at lam={xs[ys.index(max(ys))]}")

    settings = rows[0]
    caption = settings_caption(
        lr=settings["lr"],
        iters=settings["iters"],
        plt=settings["plt"],
        dt=DT,
        l2="on" if settings["l2"] == "1" else "off",
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


def control_plots(
    params,
    ic=[0, 1, 1.05],
    lr=0.05,
    train_lyapunov_times=1.0,
    plot_lyapunov_times=25,
    iters=600,
    regularized=False,
    save=False,
    lam=LAMBDA,
    integrator=euler_step,
    pts=None,
    us=None,
):
    # a caller that batched the rollout already holds the trajectory, and
    # re-integrating it one lambda at a time would undo the point of batching
    if pts is None:
        steps = round(plot_lyapunov_times / (LYAPUNOV_EXP * DT))
        pts, us = control_trajectory(
            params, initial=ic, steps=steps, integrator=integrator
        )

    success = success_fraction(pts)
    # sweep.py scrapes this line, so keep the prefix stable
    print(f"success (fraction of time x > 0): {success:.6f}")

    caption = settings_caption(
        ic=f"({','.join(str(v) for v in ic)})",
        lr=lr,
        tlt=train_lyapunov_times,
        plt=plot_lyapunov_times,
        iters=iters,
        lam=lam,
        dt=DT,
        l2="on" if regularized else "off",
        rk4="on" if integrator is rk4_step else "off",
        success=f"{success:.3f}",
    )

    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=(1.1, 1))
    ax_attractor = fig.add_subplot(gs[:, 0], projection="3d")
    ax_traj = fig.add_subplot(gs[0, 1])
    ax_u = fig.add_subplot(gs[1, 1], sharex=ax_traj)

    plot_control(pts, us, regularized=regularized, axes=(ax_traj, ax_u))
    plot_control_attractor(pts, us, regularized=regularized, ax=ax_attractor)

    fig.tight_layout(rect=(0, 0.03, 1, 1) if caption else None)
    add_caption(fig, caption)

    if save:
        path = f"./plots/loss_sweep_{iters}/attractor/"
        filename = f"lambda{lam}_{train_lyapunov_times}"

        if regularized:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ic",
        "--initial_condition",
        nargs=3,
        type=float,
        default=[0, 1, 1.05],
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.05)
    parser.add_argument("-tlt", "--train_lyapunov_times", type=float, default=1)
    parser.add_argument("-plt", "--plot_lyapunov_times", type=float, default=100)
    parser.add_argument("-i", "--iters", type=int, default=600)
    parser.add_argument("-lam", "--tuning_rate", type=float, default=0.07)
    parser.add_argument("-gg", "--gradient_growth", type=float, default=0)
    parser.add_argument(
        "-sc",
        "--success_csv",
        nargs="?",
        const="./plots/loss_sweep_1000/success.csv",
        default=None,
        help="plot success vs lambda from a sweep.py csv instead of training",
    )

    flags = parser.add_argument_group(title="Flags")
    flags.add_argument("-s", "--save", action="store_true")
    flags.add_argument("-l2", "--l2_regularized", action="store_true")
    flags.add_argument("-rk4", "--rk4", action="store_true")
    flags.add_argument("-loss", "--loss_curve", action="store_true")
    args = parser.parse_args()

    if args.success_csv:
        plot_success_vs_lambda(args.success_csv, save=args.save)
    elif args.gradient_growth:
        plot_gradient_vs_window(
            ic=args.initial_condition,
            max_lt=args.gradient_growth,
            integrator=rk4_step if args.rk4 else euler_step,
        )
    else:
        integrator = rk4_step if args.rk4 else euler_step

        # trained once here, then shared by both plots
        history = []
        params = optimize_gradient(
            ic=args.initial_condition,
            lr=args.learning_rate,
            lyapunov_times=args.train_lyapunov_times,
            iters=args.iters,
            regularized=args.l2_regularized,
            lam=args.tuning_rate,
            integrator=integrator,
            history=history,
        )

        if args.loss_curve:
            plot_loss_curve(
                history,
                ic=args.initial_condition,
                lr=args.learning_rate,
                train_lyapunov_times=args.train_lyapunov_times,
                iters=args.iters,
                regularized=args.l2_regularized,
                lam=args.tuning_rate,
                integrator=integrator,
                save=args.save,
            )

        control_plots(
            params,
            ic=args.initial_condition,
            lr=args.learning_rate,
            train_lyapunov_times=args.train_lyapunov_times,
            plot_lyapunov_times=args.plot_lyapunov_times,
            iters=args.iters,
            regularized=args.l2_regularized,
            save=args.save,
            lam=args.tuning_rate,
            integrator=integrator,
        )
