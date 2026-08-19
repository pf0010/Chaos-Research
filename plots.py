from sim import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import argparse


def plot_attractor(points_sets):
    ax = plt.figure().add_subplot(projection="3d")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")

    for p in points_sets:
        ax.plot(*p.T, lw=0.5)


def plot_x_vs_t(points_sets):
    fig = plt.figure().add_subplot()
    fig.axhline(0, color="black", linestyle="--", linewidth=1)
    lyapunov_times = np.arange(len(points_sets[0].T[0])) * DT * LYAPUNOV_EXP

    for i, p in enumerate(points_sets):
        xs = p.T[0]
        fig.plot(lyapunov_times, xs, linestyle="-", linewidth=0.5, label=f"p{i + 1}")

    fig.set_xlabel("Lyapunov times (t / τ)")
    fig.set_ylabel("x")
    fig.legend()


def plot_phi_vs_t(points_sets, eps=EPS):
    fig = plt.figure().add_subplot()
    fig.axhline(0.5, color="black", linestyle="--", linewidth=1)
    lyapunov_times = np.arange(len(points_sets[0].T[0])) * DT * LYAPUNOV_EXP

    for i, p in enumerate(points_sets):
        fig.plot(
            lyapunov_times,
            phi(p.T[0], eps),
            linestyle="-",
            linewidth=0.5,
            label=f"p{i + 1}",
        )

    fig.set_ylim(-0.05, 1.05)
    fig.set_xlabel("Lyapunov times (t / τ)")
    fig.set_ylabel("φ")
    fig.set_title(f"Soft lobe indicator (ε = {eps})")
    fig.legend()


def plot_gradient_separation(norms):
    fig = plt.figure().add_subplot()
    fig.semilogy(
        np.arange(len(norms)) * DT,
        norms,
        label="Exponential Growth",
        color="blue",
        linewidth=0.5,
    )

    fig.set_xlabel("time")
    fig.set_ylabel("separation ‖∇1 − ∇2‖")
    fig.set_title("Gradient divergence")


def controller_actions(points, params):
    w, b = params
    pts = (
        points
        if isinstance(points, torch.Tensor)
        else torch.as_tensor(np.asarray(points))
    )
    return (pts.detach().to(torch.float64) @ w.detach() + b.detach()).numpy()


def plot_control(params, initial=(0, 1, 1.05), steps=300, regularized=False):
    # COLORS = ("green", "orange", "black")
    traj = tensor_data(*initial, lambda s: control_force(s, params), steps=steps)
    pts = traj.detach().numpy()
    us = controller_actions(traj, params)
    lyapunov_times = np.arange(len(pts)) * DT * LYAPUNOV_EXP

    fig, (ax_traj, ax_u) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))

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

    fig.tight_layout()


def plot_control_attractor(params, initial=(0, 1, 1.05), steps=300, regularized=False):
    traj = tensor_data(*initial, lambda s: control_force(s, params), steps=steps)
    pts = traj.detach().numpy()
    us = controller_actions(traj, params)

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


def plot_gradient_vs_window(ic=(0, 1, 1.05), max_lt=8.0, n=50, save=False):
    windows = np.linspace(0.1, max_lt, n)
    norms = []
    for lt in windows:
        params = create_controller()
        steps = round(lt / (LYAPUNOV_EXP * DT))
        traj = tensor_data(*ic, lambda s: control_force(s, params), steps=steps)
        g = torch.autograd.grad(calculate_loss(traj), params)
        norms.append(torch.cat([gi.reshape(-1) for gi in g]).norm().item())

    ax = plt.figure().add_subplot()
    ax.semilogy(windows, norms, color="red", linewidth=0.8)
    ax.set_xlabel("Lyapunov times (t / τ)")
    ax.set_ylabel("Loss-gradient")
    ax.set_title("Loss-gradient blowup vs. training window")

    if save:
        plt.savefig(
            f"./plots/gradient_growth/{int(max_lt)}.png", dpi=150, bbox_inches="tight"
        )
        print(f"Saved to ./plots/gradient_growth/{int(max_lt)}.png")
    else:
        plt.show()


def lorenz_plots():
    p1, p1_grad = array_data(0, 1, 1.05)
    p2, p2_grad = array_data(0, 1, 1.05, u=1)

    points = [p1, p2]
    norms = np.linalg.norm(p1_grad - p2_grad, axis=1)

    # plot_attractor(points)
    # plot_x_vs_t(points)
    plot_phi_vs_t(points)
    # plot_gradient_separation(norms)
    plt.show()


def control_plots(
    ic=[0, 1, 1.05],
    lr=0.05,
    train_lyapunov_times=1.0,
    plot_lyapunov_times=25,
    iters=600,
    regularized=False,
    save=False,
    lam=LAMBDA
):
    params = optimize_gradient(
        ic=ic,
        lr=lr,
        lyapunov_times=train_lyapunov_times,
        iters=iters,
        regularized=regularized,
        lam=lam
    )

    steps = round(plot_lyapunov_times / (LYAPUNOV_EXP * DT))

    if save:
        path = f"./plots/control/lambda_{lam}/"
        filename = f"{train_lyapunov_times}"

        if regularized:
            filename += "_regularized"
        else:
            filename += "_unregularized"

        plot_control(params, steps=steps, regularized=regularized)
        plt.savefig(path + filename + "_traj.png", dpi=150, bbox_inches="tight")
        plot_control_attractor(params, steps=steps, regularized=regularized)
        plt.savefig(path + filename + "_attractor.png", dpi=150, bbox_inches="tight")
    else:
        plot_control(params, steps=steps, regularized=regularized)
        plot_control_attractor(params, steps=steps, regularized=regularized)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ic", "--initial_condition", nargs=3, type=float, default=[0, 1, 1.05]
    )
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.05)
    parser.add_argument("-tlt", "--train_lyapunov_times", type=float, default=1)
    parser.add_argument("-plt", "--plot_lyapunov_times", type=float, default=100)
    parser.add_argument("-i", "--iters", type=int, default=600)
    parser.add_argument("-l2", "--l2_regularized", action="store_true")
    parser.add_argument("-gg", "--gradient_growth", type=float, default=0)
    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("-lam", "--tuning_rate", type=float, default=0.07)
    args = parser.parse_args()

    if args.gradient_growth:
        plot_gradient_vs_window(ic=args.initial_condition, max_lt=args.gradient_growth)
    else:
        control_plots(
            ic=args.initial_condition,
            lr=args.learning_rate,
            train_lyapunov_times=args.train_lyapunov_times,
            plot_lyapunov_times=args.plot_lyapunov_times,
            iters=args.iters,
            regularized=args.l2_regularized,
            save=args.save,
            lam=args.tuning_rate
        )
