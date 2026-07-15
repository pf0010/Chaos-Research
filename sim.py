import numpy as np
import torch

RAYLEIGH = 28
PRANDTL = 10
B = 8 / 3
TIMESTEPS = 5000
DT = 0.01
LYAPUNOV_EXP = 0.9056
EPS = 2.0

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

def create_controller():
    w = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)

    return w, b

def control_force(state, params):
    w, b = params
    return torch.dot(w, state) + b

def tensor_data(initial_x, initial_y, initial_z, control, steps=TIMESTEPS):
    state = torch.tensor([initial_x, initial_y, initial_z], dtype=torch.float64)

    points = [state]

    for _ in range(steps):
        x, y, z = state

        ut = control(state) if callable(control) else control

        dx = PRANDTL * (y - x) + ut
        dy = x * (RAYLEIGH - z) - y
        dz = x * y - B * z

        state = state + torch.stack((dx, dy, dz)) * DT

        points.append(state)

    return torch.stack(points)

def position_gradient(initial_x, initial_y, initial_z, ut, lyapunov_times=1.0, coord=0):
    steps = round(lyapunov_times / (LYAPUNOV_EXP * DT))
    traj = tensor_data(initial_x, initial_y, initial_z, ut, steps=steps)

    grad, = torch.autograd.grad(traj[steps][coord], ut)

    return traj[steps].detach().numpy(), grad.item()

def phi(x, eps=EPS):
    tanh = torch.tanh if isinstance(x, torch.Tensor) else np.tanh

    return 0.5 * (1 + tanh(x / eps))

def calculate_loss(p, eps=EPS):
    x = p[:, 0]
    # loss = (torch.relu(-x) ** 2).mean()

    return phi(x, eps).mean()

def optimize_gradient(params=None, lyapunov_times=1.0, iters=600, lr=0.1):
    steps = round(lyapunov_times / (LYAPUNOV_EXP * DT))

    if params is None:
        params = create_controller()

    opt = torch.optim.Adam(params, lr=lr)

    for i in range(iters):
        opt.zero_grad()
        traj = tensor_data(0, 1, 1.05, lambda s: control_force(s, params), steps=steps)
        loss = calculate_loss(traj)
        loss.backward()
        opt.step()

        if i % 20 == 0:
            w, b = params
            print(f"iter {i:4d}  loss {loss.item():.4f}   "
                  f"w {w.detach().numpy().round(3)}   b {b.item():+.3f}")

    return params


if __name__ == "__main__":
    w, b = optimize_gradient(lr=0.02)
    print("learned feedback law: w =", w.detach().numpy(), ". (x,y,z) +", b.item())
