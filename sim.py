import numpy as np
import torch

RAYLEIGH = 28
PRANDTL = 10
B = 8 / 3
TIMESTEPS = 5000
DT = 0.005
LYAPUNOV_EXP = 0.9056
EPS = 2.0
U_REF = 60.0
LAMBDA = 0.07


def lorenz(state, u=0.0):
    x, y, z = state[0], state[1], state[2]

    dx = PRANDTL * (y - x) + u
    dy = x * (RAYLEIGH - z) - y
    dz = x * y - B * z

    if isinstance(state, torch.Tensor):
        return torch.stack((dx, dy, dz))

    return np.array((dx, dy, dz))


def control_input(control, state):
    return control(state) if callable(control) else control


def euler_step(state, control, dt=DT):
    k1 = lorenz(state, control_input(control, state))

    return state + dt * k1, k1


def rk4_step(state, control, dt=DT):
    # the control is re-evaluated at every stage, so this integrates the
    # closed-loop field rather than holding u fixed across the step
    k1 = lorenz(state, control_input(control, state))

    s2 = state + 0.5 * dt * k1
    k2 = lorenz(s2, control_input(control, s2))

    s3 = state + 0.5 * dt * k2
    k3 = lorenz(s3, control_input(control, s3))

    s4 = state + dt * k3
    k4 = lorenz(s4, control_input(control, s4))

    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), k1


def array_data(
    initial_x, initial_y, initial_z, steps=TIMESTEPS, u=0.0, integrator=euler_step
):
    state = np.array((initial_x, initial_y, initial_z), dtype=float)

    points = np.empty((steps + 1, 3))
    points[0] = state

    gradient = np.empty((steps + 1, 3))

    for t in range(steps):
        state, k1 = integrator(state, u)

        gradient[t] = k1
        points[t + 1] = state

    gradient[steps] = lorenz(state, u)

    return points, gradient


def create_controller():
    w = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)

    return w, b


def control_force(state, params):
    w, b = params
    return torch.dot(w, state) + b


def tensor_data(
    initial_x, initial_y, initial_z, control, steps=TIMESTEPS, integrator=euler_step
):
    state = torch.tensor([initial_x, initial_y, initial_z], dtype=torch.float64)

    points = [state]

    for _ in range(steps):
        state, _ = integrator(state, control)

        points.append(state)

    return torch.stack(points)


def position_gradient(initial_x, initial_y, initial_z, ut, lyapunov_times=1.0, coord=0):
    steps = round(lyapunov_times / (LYAPUNOV_EXP * DT))
    traj = tensor_data(initial_x, initial_y, initial_z, ut, steps=steps)

    (grad,) = torch.autograd.grad(traj[steps][coord], ut)

    return traj[steps].detach().numpy(), grad.item()


def phi(x, eps=EPS):
    tanh = torch.tanh if isinstance(x, torch.Tensor) else np.tanh

    return 0.5 * (1 + tanh(x / eps))


def calculate_loss(p, eps=EPS):
    x = p[:, 0]

    return (1 - phi(x, eps)).mean()


def success_fraction(p):
    # hard counterpart of calculate_loss: no tanh softening, so this is the
    # metric we actually care about rather than the one we differentiate
    x = p[:, 0]

    if isinstance(x, torch.Tensor):
        return (x > 0).to(torch.float64).mean().item()

    return float((x > 0).mean())


def control_effort(p, params, u_ref=U_REF):
    w, b = params

    # the control at the final state is never applied, so drop it
    u = p[:-1] @ w + b

    return (u / u_ref).pow(2).mean()


def optimize_gradient(
    ic=[0, 1, 1.05],
    params=None,
    lyapunov_times=1.0,
    iters=600,
    lr=0.1,
    lam=LAMBDA,
    regularized=False,
    integrator=euler_step,
    history=None,
    verbose=True,
):
    # history, if given, is a list that collects (task, penalty, total) per iter
    steps = round(lyapunov_times / (LYAPUNOV_EXP * DT))

    if params is None:
        params = create_controller()

    opt = torch.optim.Adam(params, lr=lr)

    for i in range(iters):
        opt.zero_grad()
        traj = tensor_data(
            *ic, lambda s: control_force(s, params), steps=steps, integrator=integrator
        )
        task = calculate_loss(traj)
        effort = control_effort(traj, params)

        if regularized:
            loss = task + lam * effort
        else:
            loss = task
        loss.backward()
        opt.step()

        if history is not None:
            history.append((task.item(), lam * effort.item(), loss.item()))

        if verbose and i % 20 == 0:
            w, b = params
            rms = U_REF * effort.item() ** 0.5
            print(
                f"iter {i:4d}  loss {loss.item():.4f}   task {task.item():.4f}   "
                f"pen {lam * effort.item():.4f}   |u|rms {rms:7.2f}   "
                f"w {w.detach().numpy().round(3)}   b {b.item():+.3f}"
            )

    return params


if __name__ == "__main__":
    w, b = optimize_gradient(lr=0.02)
    print("learned feedback law: w =", w.detach().numpy(), ". (x,y,z) +", b.item())
