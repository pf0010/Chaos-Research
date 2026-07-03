import matplotlib.pyplot as plt
import numpy as np
import torch

RAYLEIGH = 28
PRANDTL = 10
B = 8 / 3
TIMESTEPS = 5000
DT = 0.01
LYAPUNOV_EXP = 0.9056

def array_data(initial_x, initial_y, initial_z, steps=TIMESTEPS, u=0.0):
    x, y, z = initial_x, initial_y, initial_z

    points = np.empty((steps + 1, 3))
    points[0] = (initial_x, initial_y, initial_z)

    gradient = np.empty((steps + 1, 3))

    for t in range(steps):
        dx = PRANDTL * (y - x) + u
        dy = x * (RAYLEIGH - z) - y
        dz = x * y - B * z

        gradient[t] = (dx, dy, dz)

        x += dx * DT
        y += dy * DT
        z += dz * DT

        points[t + 1] = (x, y, z)

    gradient[steps] = (PRANDTL * (y - x) + u, x * (RAYLEIGH - z) - y, x * y - B * z)

    return points, gradient

def tensor_data(initial_x, initial_y, initial_z, ut, steps=TIMESTEPS):
    x = torch.tensor(initial_x, dtype=torch.float64)
    y = torch.tensor(initial_y, dtype=torch.float64)
    z = torch.tensor(initial_z, dtype=torch.float64)

    points = [torch.stack((x, y, z))]

    for _ in range(steps):
        dx = PRANDTL * (y - x) + ut
        dy = x * (RAYLEIGH - z) - y
        dz = x * y - B * z

        x = x + dx * DT
        y = y + dy * DT
        z = z + dz * DT

        points.append(torch.stack((x, y, z)))

    return torch.stack(points)

def position_gradient(initial_x, initial_y, initial_z, ut, lyapunov_times=1.0, coord=0):
    steps = round(lyapunov_times / (LYAPUNOV_EXP * DT))
    traj = tensor_data(initial_x, initial_y, initial_z, ut, steps=steps)

    grad, = torch.autograd.grad(traj[steps][coord], ut)

    return traj[steps].detach().numpy(), grad.item()

def calculate_loss(p):
    x = p[:, 0]
    loss = (torch.relu(-x) ** 2).mean()

    return loss

def optimize_gradient(u, lyapunov_times=1.0):
    ITERS = 600
    steps = round(lyapunov_times / (LYAPUNOV_EXP * DT))   # horizon, kept short
    opt = torch.optim.Adam([u], lr=0.1)

    for i in range(ITERS):
        opt.zero_grad()
        traj = tensor_data(0, 1, 1.05, u, steps=steps)
        loss = calculate_loss(traj)
        loss.backward()
        opt.step()

        if i % 20 == 0:
            print(f"iter {i:4d}  loss {loss.item():.4f}  u {u.item():+.4f}")

    return u.item()


if __name__ == "__main__":
    p1u = 0
    # p2u = 1
    u1 = torch.tensor(1, requires_grad=True, dtype=torch.float64)
    optimize_gradient(u1)
