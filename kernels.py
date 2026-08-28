"""The CUDA fast path for the undifferentiated moment rollout.

`effort_moments` is the single most expensive thing in a training iteration:
it integrates the *evaluation* window, tens of thousands of steps, once per
iteration. It is also the easiest thing here to hand to a GPU, because it is
deliberately not differentiated -- the gradient reaches w and b analytically
through `effort_from_moments`, so this side needs no backward pass at all.

The catch is that the state is three numbers. Written as torch ops, a step is
~10 kernel launches over 3-element tensors, and at ~5 us of launch overhead
each the GPU is 2.5x *slower* than the CPU no matter how big the batch gets --
it is never computing, only launching. So the whole time loop goes inside one
kernel instead: one thread per grid point, x/y/z and the moment accumulators
live in registers for the entire rollout, and there is exactly one launch and
one write at the end. Measured on an RTX 5070, 22085 steps:

    B       torch/CPU     torch/CUDA     this kernel (fp32)
    1          428 ms        1208 ms               0.38 ms
    55         483 ms        1245 ms               0.38 ms
    4096      3992 ms        1345 ms               0.38 ms

Falls back to `moment_stats_numpy` when there is no CUDA device, which is the
same batched arithmetic and still beats looping over the grid one point at a
time.
"""

import numpy as np
import torch

from lorenz import BETA, DT, PRANDTL, RAYLEIGH, euler_step, rk4_step

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - triton ships with the cuda wheels
    HAVE_TRITON = False


def cuda_available():
    return HAVE_TRITON and torch.cuda.is_available()


def resolve_device(device=None):
    """'auto' (or None) picks CUDA when there is one, and says so nowhere else."""
    if device in (None, "auto"):
        return "cuda" if cuda_available() else "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda was asked for but torch sees no CUDA device")

    return device


def default_dtype(device):
    """float32 on the GPU, float64 on the CPU.

    fp64 is 1/64 rate on a consumer card and costs ~37x in this kernel, and it
    buys nothing measurable: over the training window the fp32 loss gradient
    agrees with fp64 to a relative 1e-7, and over the evaluation window the
    moments are ergodic averages whose value moves ~4% under a 1e-12 nudge to
    the initial condition -- an order of magnitude more than the ~1% fp32 costs.
    On the CPU fp64 is free, so nothing is given up by keeping it there.
    """
    return torch.float32 if device == "cuda" else torch.float64


if HAVE_TRITON:
    _RAYLEIGH = tl.constexpr(float(RAYLEIGH))
    _PRANDTL = tl.constexpr(float(PRANDTL))
    _BETA = tl.constexpr(float(BETA))
    _DT = tl.constexpr(float(DT))

    @triton.jit
    def _rhs(x, y, z, u):
        return (
            _PRANDTL * (y - x) + u,
            x * (_RAYLEIGH - z) - y,
            x * y - _BETA * z,
        )

    @triton.jit
    def _moments_kernel(
        W, Bv, S0, Mout, mout, n_batch, steps, RK4: tl.constexpr, BLOCK: tl.constexpr
    ):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n_batch

        w0 = tl.load(W + off * 3 + 0, mask=mask, other=0.0)
        w1 = tl.load(W + off * 3 + 1, mask=mask, other=0.0)
        w2 = tl.load(W + off * 3 + 2, mask=mask, other=0.0)
        b = tl.load(Bv + off, mask=mask, other=0.0)

        x = tl.load(S0 + off * 3 + 0, mask=mask, other=0.0)
        y = tl.load(S0 + off * 3 + 1, mask=mask, other=0.0)
        z = tl.load(S0 + off * 3 + 2, mask=mask, other=0.0)

        mx = tl.zeros_like(x)
        my = tl.zeros_like(x)
        mz = tl.zeros_like(x)
        Mxx = tl.zeros_like(x)
        Mxy = tl.zeros_like(x)
        Mxz = tl.zeros_like(x)
        Myy = tl.zeros_like(x)
        Myz = tl.zeros_like(x)
        Mzz = tl.zeros_like(x)

        for _ in range(steps):
            # accumulated before the step, so the final state is dropped -- the
            # same window `effort_penalty`'s traj[:-1] takes, and for the same
            # reason: that control is never applied
            mx += x
            my += y
            mz += z
            Mxx += x * x
            Mxy += x * y
            Mxz += x * z
            Myy += y * y
            Myz += y * z
            Mzz += z * z

            u = w0 * x + w1 * y + w2 * z + b
            k1x, k1y, k1z = _rhs(x, y, z, u)

            if RK4:
                # the control is re-evaluated at every stage, matching rk4_step
                x2 = x + 0.5 * _DT * k1x
                y2 = y + 0.5 * _DT * k1y
                z2 = z + 0.5 * _DT * k1z
                k2x, k2y, k2z = _rhs(x2, y2, z2, w0 * x2 + w1 * y2 + w2 * z2 + b)

                x3 = x + 0.5 * _DT * k2x
                y3 = y + 0.5 * _DT * k2y
                z3 = z + 0.5 * _DT * k2z
                k3x, k3y, k3z = _rhs(x3, y3, z3, w0 * x3 + w1 * y3 + w2 * z3 + b)

                x4 = x + _DT * k3x
                y4 = y + _DT * k3y
                z4 = z + _DT * k3z
                k4x, k4y, k4z = _rhs(x4, y4, z4, w0 * x4 + w1 * y4 + w2 * z4 + b)

                x += (_DT / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
                y += (_DT / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
                z += (_DT / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z)
            else:
                x += _DT * k1x
                y += _DT * k1y
                z += _DT * k1z

        n = steps

        tl.store(mout + off * 3 + 0, mx / n, mask=mask)
        tl.store(mout + off * 3 + 1, my / n, mask=mask)
        tl.store(mout + off * 3 + 2, mz / n, mask=mask)

        tl.store(Mout + off * 9 + 0, Mxx / n, mask=mask)
        tl.store(Mout + off * 9 + 1, Mxy / n, mask=mask)
        tl.store(Mout + off * 9 + 2, Mxz / n, mask=mask)
        tl.store(Mout + off * 9 + 3, Mxy / n, mask=mask)
        tl.store(Mout + off * 9 + 4, Myy / n, mask=mask)
        tl.store(Mout + off * 9 + 5, Myz / n, mask=mask)
        tl.store(Mout + off * 9 + 6, Mxz / n, mask=mask)
        tl.store(Mout + off * 9 + 7, Myz / n, mask=mask)
        tl.store(Mout + off * 9 + 8, Mzz / n, mask=mask)


BLOCK = 64


def moment_stats_cuda(state0, w, b, steps, integrator=euler_step):
    """<s sᵀ> and <s> over `steps`, for a batch of policies, in one launch."""
    w = w.contiguous()
    b = b.contiguous()
    # a caller that already holds the batch passes a tensor; broadcast anything
    # else up to it, so one shared initial condition needs no ceremony
    if not isinstance(state0, torch.Tensor):
        state0 = torch.as_tensor(
            np.asarray(state0, dtype=float), device=w.device, dtype=w.dtype
        )

    state0 = state0.to(device=w.device, dtype=w.dtype).expand_as(w).contiguous()
    n_batch = w.shape[0]

    M = torch.empty(n_batch, 3, 3, device=w.device, dtype=w.dtype)
    m = torch.empty(n_batch, 3, device=w.device, dtype=w.dtype)

    _moments_kernel[(triton.cdiv(n_batch, BLOCK),)](
        w,
        b,
        state0,
        M,
        m,
        n_batch,
        steps,
        RK4=integrator is rk4_step,
        BLOCK=BLOCK,
        num_warps=2,
    )

    return M, m


def moment_stats_numpy(state0, w, b, steps, integrator=euler_step):
    """The CPU path: the same accumulation, batched over the leading axis."""
    w = np.asarray(w, dtype=float)
    b = np.asarray(b, dtype=float)
    state = np.broadcast_to(np.asarray(state0, dtype=float), w.shape).copy()

    def control(states):
        return (states * w).sum(-1) + b

    m = np.zeros_like(state)
    M = np.zeros((*w.shape, 3))

    for _ in range(steps):
        m += state
        M += state[..., :, None] * state[..., None, :]

        state, _ = integrator(state, control)

    return M / steps, m / steps


def moment_stats(state0, w, b, steps, integrator=euler_step):
    """Dispatch on where the parameters already live."""
    if isinstance(w, torch.Tensor) and w.is_cuda:
        return moment_stats_cuda(state0, w, b, steps, integrator)

    def to_numpy(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

    M, m = moment_stats_numpy(
        to_numpy(state0), to_numpy(w), to_numpy(b), steps, integrator
    )

    if isinstance(w, torch.Tensor):
        return (
            torch.as_tensor(M, dtype=w.dtype, device=w.device),
            torch.as_tensor(m, dtype=w.dtype, device=w.device),
        )

    return M, m
