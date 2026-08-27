"""The Lorenz system, its integrators, and trajectory rollouts.

Knows nothing about controllers beyond the fact that a step takes some `u`:
`eval_control` accepts either a constant or a callable of the state, so the
same integrators serve an open-loop forcing and a learned feedback law.
"""

import numpy as np
import torch

RAYLEIGH = 28
PRANDTL = 10
BETA = 8 / 3
DEFAULT_STEPS = 5000
DT = 0.005
LYAPUNOV_EXP = 0.9056


def lorenz_rhs(state, u=0.0):
    x, y, z = state[0], state[1], state[2]

    dx = PRANDTL * (y - x) + u
    dy = x * (RAYLEIGH - z) - y
    dz = x * y - BETA * z

    if isinstance(state, torch.Tensor):
        return torch.stack((dx, dy, dz))

    return np.array((dx, dy, dz))


def eval_control(control, state):
    return control(state) if callable(control) else control


def euler_step(state, control, dt=DT):
    k1 = lorenz_rhs(state, eval_control(control, state))

    return state + dt * k1, k1


def rk4_step(state, control, dt=DT):
    # the control is re-evaluated at every stage, so this integrates the
    # closed-loop field rather than holding u fixed across the step
    k1 = lorenz_rhs(state, eval_control(control, state))

    s2 = state + 0.5 * dt * k1
    k2 = lorenz_rhs(s2, eval_control(control, s2))

    s3 = state + 0.5 * dt * k2
    k3 = lorenz_rhs(s3, eval_control(control, s3))

    s4 = state + dt * k3
    k4 = lorenz_rhs(s4, eval_control(control, s4))

    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), k1


def rollout_numpy(state0, steps=DEFAULT_STEPS, u=0.0, integrator=euler_step):
    state = np.array(state0, dtype=float)

    traj = np.empty((steps + 1, 3))
    traj[0] = state

    derivs = np.empty((steps + 1, 3))

    for t in range(steps):
        state, k1 = integrator(state, u)

        derivs[t] = k1
        traj[t + 1] = state

    derivs[steps] = lorenz_rhs(state, u)

    return traj, derivs


def rollout_torch(state0, control, steps=DEFAULT_STEPS, integrator=euler_step):
    state = torch.as_tensor(state0, dtype=torch.float64)

    traj = [state]

    for _ in range(steps):
        state, _ = integrator(state, control)

        traj.append(state)

    return torch.stack(traj)
