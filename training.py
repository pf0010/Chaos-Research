"""The learned controller: policy, objectives, and the training loop.

The policy is a linear feedback law u = w·s + b, trained by differentiating
straight through the integrator, so the training window is limited by how far
the gradient survives the attractor's exponential stretching.
"""

import numpy as np
import torch

from lorenz import DT, LYAPUNOV_EXP, euler_step, rollout_torch
from params import DEFAULT_EFFORT_WEIGHT

SOFTNESS = 2.0
U_REF = 60.0


def init_policy_params():
    w = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)

    return w, b


def linear_policy(params, state):
    w, b = params
    return torch.dot(w, state) + b


def final_state_sensitivity(state0, u_tensor, horizon=1.0, coord=0):
    steps = round(horizon / (LYAPUNOV_EXP * DT))
    traj = rollout_torch(state0, u_tensor, steps=steps)

    (grad,) = torch.autograd.grad(traj[steps][coord], u_tensor)

    return traj[steps].detach().numpy(), grad.item()


def soft_step(x, softness=SOFTNESS):
    tanh = torch.tanh if isinstance(x, torch.Tensor) else np.tanh

    return 0.5 * (1 + tanh(x / softness))


def task_loss(traj, softness=SOFTNESS):
    x = traj[:, 0]

    return (1 - soft_step(x, softness)).mean()


def success_fraction(traj):
    # hard counterpart of task_loss: no tanh softening, so this is the
    # metric we actually care about rather than the one we differentiate
    x = traj[:, 0]

    if isinstance(x, torch.Tensor):
        return (x > 0).to(torch.float64).mean().item()

    return float((x > 0).mean())


def effort_penalty(traj, params, u_ref=U_REF):
    w, b = params

    # the control at the final state is never applied, so drop it
    u = traj[:-1] @ w + b

    return (u / u_ref).pow(2).mean()


def train_policy(
    state0=[0, 1, 1.05],
    params=None,
    horizon=1.0,
    iters=600,
    lr=0.1,
    effort_weight=DEFAULT_EFFORT_WEIGHT,
    penalize_effort=False,
    integrator=euler_step,
    history=None,
    verbose=True,
):
    # history, if given, is a list that collects (task, penalty, total) per iter
    steps = round(horizon / (LYAPUNOV_EXP * DT))

    if params is None:
        params = init_policy_params()

    opt = torch.optim.Adam(params, lr=lr)

    for i in range(iters):
        opt.zero_grad()
        traj = rollout_torch(
            state0,
            lambda s: linear_policy(params, s),
            steps=steps,
            integrator=integrator,
        )
        task = task_loss(traj)
        effort = effort_penalty(traj, params)

        if penalize_effort:
            loss = task + effort_weight * effort
        else:
            loss = task
        loss.backward()
        opt.step()

        if history is not None:
            history.append((task.item(), effort_weight * effort.item(), loss.item()))

        if verbose and i % 20 == 0:
            w, b = params
            rms = U_REF * effort.item() ** 0.5
            print(
                f"iter {i:4d}  loss {loss.item():.4f}   task {task.item():.4f}   "
                f"pen {effort_weight * effort.item():.4f}   |u|rms {rms:7.2f}   "
                f"w {w.detach().numpy().round(3)}   b {b.item():+.3f}"
            )

    return params


if __name__ == "__main__":
    w, b = train_policy(lr=0.02)
    print("learned feedback law: w =", w.detach().numpy(), ". (x,y,z) +", b.item())
