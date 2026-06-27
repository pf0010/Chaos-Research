import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

RAYLEIGH = 28
PRANDTL = 10
B = 8 / 3
TIMESTEPS = 5000
DT = 0.01
LYAPUNOV_EXP = 0.9056  # largest Lyapunov exponent for the classic Lorenz params

def data(initial_x, initial_y, initial_z):
    dx, dy, dz = 0, 0, 0
    x, y, z = initial_x, initial_y, initial_z

    points = np.empty((TIMESTEPS + 1, 3))
    points[0] = (initial_x, initial_y, initial_z)

    gradient = np.empty((TIMESTEPS + 1, 3))
    gradient[0] = (dx, dy, dz)

    for t in range(TIMESTEPS):
        dx = PRANDTL * (y - x)
        dy = x * (RAYLEIGH - z) - y
        dz = x * y - B * z

        x += dx * DT
        y += dy * DT
        z += dz * DT

        points[t + 1] = (np.array([x, y, z]))
        gradient[t + 1] = (np.array([dx, dy, dz]))

    return points, gradient

p1, p1_grad = data(0, 1, 1.05)
p2, p2_grad = data(0.000001, 1, 1.05)
lyapunov_times = np.arange(len(p1.T[0])) * DT * LYAPUNOV_EXP
norms = np.empty((TIMESTEPS + 1))

for t, (i, j) in enumerate(zip(p1_grad, p2_grad)):
    dx1, dy1, dz1 = i
    dx2, dy2, dz2 = j

    norm = sqrt((dx1 - dx2) ** 2 + (dy1 - dy2) ** 2 + (dz1 - dz2) ** 2)
    norms[t] = norm

def plot_attractor(points_sets):
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title("Lorenz Attractor")

    for p in points_sets:
        ax.plot(*p.T, lw=0.5)

def plot_x_vs_t():
    x1s = p1.T[0]
    x2s = p2.T[0]

    fig = plt.figure().add_subplot()
    fig.plot(lyapunov_times, x1s, linestyle='-', color='b', linewidth=0.5)
    fig.plot(lyapunov_times, x2s, linestyle='-', color='r', linewidth=0.5)

    fig.set_xlabel("Lyapunov times (t / τ)")
    fig.set_ylabel("x")

def plot_gradient_separation():
    fig = plt.figure().add_subplot()
    fig.semilogy(np.arange(len(p1.T[0])) * DT, norms, label="Exponential Growth", color="blue", linewidth=0.5)

    # fig.grid(True, which="both", linestyle="--") # Shows major and minor grid lines
    fig.set_xlabel("time")
    fig.set_ylabel("separation ‖∇1 − ∇2‖")
    fig.set_title("Gradient divergence")

# plot_attractor([p1, p2])
# plot_x_vs_t()
# plot_gradient_separation()
plt.show()
