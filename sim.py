import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

RAYLEIGH = 28
PRANDTL = 10
B = 8 / 3
TIMESTEPS = 5000
DT = 0.01
LYAPUNOV_EXP = 0.9056

def data(initial_x, initial_y, initial_z, steps=TIMESTEPS):
    x, y, z = initial_x, initial_y, initial_z

    points = np.empty((steps + 1, 3))
    points[0] = (initial_x, initial_y, initial_z)

    gradient = np.empty((steps + 1, 3))

    for t in range(steps):
        dx = PRANDTL * (y - x)
        dy = x * (RAYLEIGH - z) - y
        dz = x * y - B * z

        gradient[t] = (dx, dy, dz)

        x += dx * DT
        y += dy * DT
        z += dz * DT

        points[t + 1] = (x, y, z)

    gradient[steps] = (PRANDTL * (y - x), x * (RAYLEIGH - z) - y, x * y - B * z)

    return points, gradient

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

    for p in points_sets:
        xs = p.T[0]
        fig.plot(lyapunov_times, xs, linestyle='-', linewidth=0.5)

    fig.set_xlabel("Lyapunov times (t / τ)")
    fig.set_ylabel("x")

def plot_gradient_separation(norms):
    fig = plt.figure().add_subplot()
    fig.semilogy(np.arange(len(p1.T[0])) * DT, norms, label="Exponential Growth", color="blue", linewidth=0.5)

    fig.set_xlabel("time")
    fig.set_ylabel("separation ‖∇1 − ∇2‖")
    fig.set_title("Gradient divergence")

if __name__ == "__main__":
    p1, p1_grad = data(0, 1, 1.05)
    p2, p2_grad = data(0.000001, 1, 1.05)
    points = [p1, p2]
    lyapunov_times = np.arange(len(p1.T[0])) * DT * LYAPUNOV_EXP
    norms = np.empty((TIMESTEPS + 1))

    norms = np.linalg.norm(p1_grad - p2_grad, axis=1)

    plot_attractor(points)
    plot_x_vs_t(points)
    plot_gradient_separation(norms)
    plt.show()
