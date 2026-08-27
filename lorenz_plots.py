import matplotlib.pyplot as plt
from sim import soft_step, rollout_numpy, DT, LYAPUNOV_EXP, SOFTNESS
import numpy as np


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


if __name__ == "__main__":
    plot_lorenz_overview()
