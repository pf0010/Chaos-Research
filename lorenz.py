"""The Lorenz system, its integrators, and trajectory rollouts.

Knows nothing about controllers beyond the fact that a step takes some `u`:
`eval_control` accepts either a constant or a callable of the state, so the
same integrators serve an open-loop forcing and a learned feedback law.

`rk4_step` is the default everywhere a rollout takes an integrator; euler is
kept because a sweep can still ask for it (`rk4=0`) and compare the two.
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
    # indexed on the last axis and stacked back onto it, so a single (3,) state
    # and a batch of them (B, 3) go through unchanged -- that is what lets one
    # rollout cover a whole sweep grid. `u` is then a scalar or a (B,) vector.
    x, y, z = state[..., 0], state[..., 1], state[..., 2]

    dx = PRANDTL * (y - x) + u
    dy = x * (RAYLEIGH - z) - y
    dz = x * y - BETA * z

    if isinstance(state, torch.Tensor):
        return torch.stack((dx, dy, dz), dim=-1)

    return np.stack((dx, dy, dz), axis=-1)


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


def rollout_numpy(state0, steps=DEFAULT_STEPS, u=0.0, integrator=rk4_step):
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


def rollout_torch(state0, control, steps=DEFAULT_STEPS, integrator=rk4_step):
    """Time-first: (steps+1, 3), or (steps+1, B, 3) if `state0` is a batch.

    A tensor `state0` is taken as it is, so the caller picks the device and the
    precision; anything else keeps the float64 CPU default the scalar runs use.
    """
    state = (
        state0
        if isinstance(state0, torch.Tensor)
        else torch.as_tensor(state0, dtype=torch.float64)
    )

    traj = [state]

    for _ in range(steps):
        state, _ = integrator(state, control)

        traj.append(state)

    return torch.stack(traj)


def rollout_numpy_batched(state0, w, b, steps=DEFAULT_STEPS, integrator=rk4_step):
    """Closed-loop rollout of B policies at once: traj (steps+1, B, 3), u (steps+1, B).

    Nothing here is differentiated -- this is the evaluation trajectory the
    figures and the success metric are read off -- so it stays in numpy, where
    a step on a (B, 3) array costs a fraction of the same step as torch ops.
    """
    w = np.asarray(w, dtype=float)
    b = np.asarray(b, dtype=float)
    state = np.broadcast_to(np.asarray(state0, dtype=float), w.shape).copy()

    def control(states):
        return (states * w).sum(-1) + b

    traj = np.empty((steps + 1, *w.shape))
    controls = np.empty((steps + 1, w.shape[0]))

    traj[0] = state
    controls[0] = control(state)

    for t in range(steps):
        state, _ = integrator(state, control)

        traj[t + 1] = state
        controls[t + 1] = control(state)

    return traj, controls
