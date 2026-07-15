from sim import *
import matplotlib.pyplot as plt

def plot_attractor(points_sets):
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title("Lorenz Attractor")

    for p in points_sets:
        ax.plot(*p.T, lw=0.5)

def plot_x_vs_t(points_sets):
    fig = plt.figure().add_subplot()
    fig.axhline(0, color='black', linestyle='--', linewidth=1)
    lyapunov_times = np.arange(len(points_sets[0].T[0])) * DT * LYAPUNOV_EXP

    for i, p in enumerate(points_sets):
        xs = p.T[0]
        fig.plot(lyapunov_times, xs, linestyle='-', linewidth=0.5, label=f"p{i + 1}")

    fig.set_xlabel("Lyapunov times (t / τ)")
    fig.set_ylabel("x")
    fig.legend()

def plot_phi_vs_t(points_sets, eps=EPS):
    fig = plt.figure().add_subplot()
    fig.axhline(0.5, color='black', linestyle='--', linewidth=1)
    lyapunov_times = np.arange(len(points_sets[0].T[0])) * DT * LYAPUNOV_EXP

    for i, p in enumerate(points_sets):
        fig.plot(lyapunov_times, phi(p.T[0], eps), linestyle='-', linewidth=0.5, label=f"p{i + 1}")

    fig.set_ylim(-0.05, 1.05)
    fig.set_xlabel("Lyapunov times (t / τ)")
    fig.set_ylabel("φ")
    fig.set_title(f"Soft lobe indicator (ε = {eps})")
    fig.legend()

def plot_gradient_separation(norms):
    fig = plt.figure().add_subplot()
    fig.semilogy(np.arange(len(norms)) * DT, norms, label="Exponential Growth", color="blue", linewidth=0.5)

    fig.set_xlabel("time")
    fig.set_ylabel("separation ‖∇1 − ∇2‖")
    fig.set_title("Gradient divergence")

def plot_gradient_growth(initial_x, initial_y, initial_z, ut, lyapunov_times=1.0, coord=0):
    horizons = np.linspace(0.1, lyapunov_times, 40)
    grads = [position_gradient(initial_x, initial_y, initial_z, ut, lyapunov_times=lt, coord=coord)[1]
             for lt in horizons]

    fig = plt.figure().add_subplot()
    fig.semilogy(horizons, np.abs(grads), color="red", linewidth=0.5)
    fig.plot(horizons, np.exp(np.asarray(horizons)), linestyle="--", color="gray",
             linewidth=0.5, label="e^(t/τ)")

    fig.set_xlabel("Lyapunov times (t / τ)")
    fig.set_ylabel("|∂x(t) / ∂u|")
    fig.set_title("Gradient growth vs. horizon")
    fig.legend()
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

if __name__ == "__main__":
    u1 = torch.tensor(1, requires_grad=True, dtype=torch.float64)

    lorenz_plots()
    plot_gradient_growth(0, 1, 1.05, u1, 15)
