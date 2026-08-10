import numpy as np
import torch

RAYLEIGH = 28
PRANDTL = 10
B = 8 / 3
TIMESTEPS = 5000
DT = 0.01
LYAPUNOV_EXP = 0.9056
EPS = 2.0
U_REF = 60.0
LAMBDA = 0.007

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

    return (1 - phi(x, eps)).mean()

def control_effort(p, params, u_ref=U_REF):
    w, b = params

    # the control at the final state is never applied, so drop it
    u = p[:-1] @ w + b

    return (u / u_ref).pow(2).mean()

def optimize_gradient(ic=[0, 1, 1.05], params=None, lyapunov_times=1.0, iters=600, lr=0.1, lam=LAMBDA,
                      regularized=False):
    steps = round(lyapunov_times / (LYAPUNOV_EXP * DT))

    if params is None:
        params = create_controller()

    opt = torch.optim.Adam(params, lr=lr)

    for i in range(iters):
        opt.zero_grad()
        traj = tensor_data(*ic, lambda s: control_force(s, params), steps=steps)
        task = calculate_loss(traj)
        effort = control_effort(traj, params)

        if regularized == "l2":
            loss = task + lam * effort
        else:
            loss = task
        loss.backward()
        opt.step()

        if i % 20 == 0:
            w, b = params
            rms = U_REF * effort.item() ** 0.5
            print(f"iter {i:4d}  loss {loss.item():.4f}   task {task.item():.4f}   "
                  f"pen {lam * effort.item():.4f}   |u|rms {rms:7.2f}   "
                  f"w {w.detach().numpy().round(3)}   b {b.item():+.3f}")

    return params


if __name__ == "__main__":
    w, b = optimize_gradient(lr=0.02)
    print("learned feedback law: w =", w.detach().numpy(), ". (x,y,z) +", b.item())
