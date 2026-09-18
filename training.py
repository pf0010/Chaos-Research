"""The learned controller: policy, objectives, and the training loop.

The policy is a linear feedback law u = w·s + b, trained by differentiating
straight through the integrator, so the training window is limited by how far
the gradient survives the attractor's exponential stretching.

That limit is why control is receding-horizon (train_receding_horizon): the
projected horizon is cut into short windows, each window trains its own law
from where the previous window's law left the system, and the sequence of
laws is the controller. The gradient only ever has to survive one window.

Everything here carries an optional leading batch axis: w is (B, 3), b is (B,),
and the objectives reduce over time only, returning one number per policy. B
independent policies then train in one set of kernels instead of B sets, which
is what makes a sweep worth putting on a GPU at all -- see kernels.py for why
a single 3-component state emphatically is not.

The effort penalty is on by default: without it nothing bounds `u`, and the
law that comes back "works" by shoving impossibly hard on x. `penalize_effort`
is kept because a run can still ask for the task term alone (`-npe`, `reg=0`)
and compare the two.
"""

from typing import NamedTuple, Optional

import numpy as np
import torch

from kernels import default_dtype, moment_stats, resolve_device
from lorenz import DT, LYAPUNOV_EXP, rk4_step, rollout_numpy_batched, rollout_torch
from params import DEFAULT_EFFORT_WEIGHT

SOFTNESS = 2
U_REF = 60.0
INIT_SCALE = 0.1


def init_policy_params(batch=None, device="cpu", dtype=torch.float64, scale=INIT_SCALE):
    """Draw w and b from N(0, scale²), independently per lane.

    Drawn in float64 on the CPU and moved afterwards, so a given torch seed
    gives the same starting law whatever device or dtype the run lands on.
    """
    shape = () if batch is None else (batch,)

    w = (scale * torch.randn((*shape, 3), dtype=torch.float64)).to(device, dtype)
    b = (scale * torch.randn(shape, dtype=torch.float64)).to(device, dtype)

    return w.requires_grad_(), b.requires_grad_()


def linear_policy(params, state):
    w, b = params

    # not torch.dot, so that a (B, 3) batch of laws applies to a (B, 3) batch
    # of states in the same expression a single law applies to a single state
    return (w * state).sum(-1) + b


def final_state_sensitivity(state0, u_tensor, horizon=1.0, coord=0):
    steps = round(horizon / (LYAPUNOV_EXP * DT))
    traj = rollout_torch(state0, u_tensor, steps=steps)

    (grad,) = torch.autograd.grad(traj[steps][coord], u_tensor)

    return traj[steps].detach().numpy(), grad.item()


def soft_step(x, softness=SOFTNESS):
    tanh = torch.tanh if isinstance(x, torch.Tensor) else np.tanh

    return 0.5 * (1 + tanh(x / softness))


def masked_mean(values, mask):
    """Mean down the time axis over the entries `mask` keeps.

    torch.where rather than a multiply by 0/1: a lane that has been integrated
    past its own window contributes exactly zero even if its tail went
    non-finite, where 0 * nan would have poisoned the whole batch's gradient.
    """
    return torch.where(mask, values, 0.0).sum(0) / mask.sum(0)


def task_loss(traj, softness=SOFTNESS, mask=None):
    # mean over the time axis only: a scalar for one policy, (B,) for a batch.
    # `mask` is how lanes with different training windows share one rollout --
    # (steps+1, B), true for the steps that lane actually trains on
    x = traj[..., 0]
    shortfall = 1 - soft_step(x, softness)

    return shortfall.mean(0) if mask is None else masked_mean(shortfall, mask)


def success_fraction(traj):
    # hard counterpart of task_loss: no tanh softening, so this is the
    # metric we actually care about rather than the one we differentiate
    x = traj[..., 0]

    if isinstance(x, torch.Tensor):
        return (x > 0).to(torch.float64).mean(0)

    return (x > 0).mean(0)


def effort_penalty(traj, params, u_ref=U_REF, mask=None):
    w, b = params

    # the control at the final state is never applied, so drop it
    u = (traj[:-1] * w).sum(-1) + b
    cost = (u / u_ref).pow(2)

    return cost.mean(0) if mask is None else masked_mean(cost, mask)


def effort_moments(state0, params, steps, integrator=rk4_step):
    """<s> and <s sᵀ> over a rollout we deliberately do not differentiate.

    The penalty is quadratic in the policy, so these two moments are the whole
    of what it needs from the trajectory -- nothing else about the states
    survives into the loss. That lets the rollout leave the autograd graph
    entirely, and keeps the memory flat in `steps` instead of storing every
    state. kernels.py picks where it actually runs.
    """
    w, b = params

    return moment_stats(state0, w.detach(), b.detach(), steps, integrator)


def effort_from_moments(moments, params, u_ref=U_REF):
    """effort_penalty on the moments: <u²> = wᵀMw + 2b mᵀw + b².

    Same value and same gradient as effort_penalty over the states those
    moments came from, since w and b are the only things it differentiates
    through -- the states are frozen either way.
    """
    M, m = (
        moment
        if isinstance(moment, torch.Tensor)
        else torch.as_tensor(moment, dtype=params[0].dtype, device=params[0].device)
        for moment in moments
    )
    w, b = params

    # batched matmul on the last two axes, so this is wᵀMw for one policy and
    # a (B,) vector of them for a batch
    quadratic = (w.unsqueeze(-2) @ M @ w.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    cross = (m * w).sum(-1)

    return (quadratic + 2 * b * cross + b * b) / u_ref**2


class BatchedAdam:
    """Adam with a per-element learning rate and a device-side step counter.

    Two departures from torch.optim.Adam, both needed here. The learning rate
    is a tensor, so `learning_rate` can be one of the axes a sweep batches over
    rather than something that forces a separate run. And the step count lives
    on the device, so the bias correction is an ordinary tensor op and the
    whole update can be captured into a CUDA graph. The arithmetic is otherwise
    exactly Adam's, and `test_batched_adam` pins it to torch's to 1e-15.
    """

    def __init__(self, params, lrs, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lrs = list(lrs)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.avg = [torch.zeros_like(p) for p in self.params]
        self.avg_sq = [torch.zeros_like(p) for p in self.params]

    def reset(self):
        """Zero the moments in place, keeping tensor identity for the graph."""
        for state in (self.avg, self.avg_sq):
            for tensor in state:
                tensor.zero_()

        self.zero_grad()

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def step(self, t):
        # t is a 1-based device tensor, not a python int, so that b**t stays a
        # recordable op instead of a host-side value baked in at capture time.
        # It has to be cast first: an integer exponent would send the power to
        # float32 by promotion, and 1 - 0.999 evaluated there is wrong in the
        # fifth digit, which is enough to move the update by ~1e-5 relative.
        t = t.to(self.avg[0].dtype)

        correction1 = 1 - self.beta1**t
        correction2 = 1 - self.beta2**t

        for p, lr, avg, avg_sq in zip(self.params, self.lrs, self.avg, self.avg_sq):
            avg.mul_(self.beta1).add_(p.grad, alpha=1 - self.beta1)
            avg_sq.mul_(self.beta2).addcmul_(p.grad, p.grad, value=1 - self.beta2)

            denom = (avg_sq / correction2).sqrt().add_(self.eps)
            p.sub_(lr * (avg / correction1) / denom)


def _as_batch(value, batch, device, dtype):
    return torch.as_tensor(
        np.broadcast_to(np.asarray(value, dtype=float), (batch,)).copy(),
        device=device,
        dtype=dtype,
    )


def train_policy_batched(
    state0=(0, 1, 1.05),
    batch=1,
    horizon=1.0,
    effort_horizon=None,
    iters=600,
    learning_rate=0.05,
    effort_weight=DEFAULT_EFFORT_WEIGHT,
    penalize_effort=True,
    integrator=rk4_step,
    device=None,
    dtype=None,
    use_graph=True,
    init=None,
    record_params=False,
):
    """Train `batch` independent policies at once.

    `state0`, `learning_rate`, `effort_weight` and `horizon` may each be a
    scalar (or single start) or a length-`batch` sequence; those are the axes a
    grid can vary within one run. The middle two are per-element numbers the
    arithmetic carries anyway, and `state0` is already a (batch, 3) tensor
    here, so one start per lane costs nothing the shared start didn't.
    `horizon` is the interesting one: it does change the step count, so the
    batch integrates to the longest window in it and each lane's loss is
    masked back to its own -- which costs max(steps) rather than sum(steps).
    The integrator, the iteration count and the evaluation window still have
    to match across a batch, so those stay separate calls.

    Each lane starts from its own random law (init_policy_params) unless
    `init` gives the starting (w, b) as a (batch, 3) and a (batch,) array.

    Returns (w, b, history, grad_norm), history being (iters, batch, 3) of
    (task, λ·effort, total) -- the same three series the scalar loop recorded --
    and grad_norm (iters, batch, 3) the 2-norms of each lane's gradient in
    (w, b) from the same three terms, taken before the optimizer step. An
    unpenalized run trains on the task alone, so its λ·effort column is zero.
    With `record_params` a fifth element follows: (iters, batch, 4) of
    (w₁, w₂, w₃, b) as they stood when each iteration's gradient was taken.
    """
    device = resolve_device(device)
    dtype = dtype if dtype is not None else default_dtype(device)

    # `horizon` may differ per lane. Every lane is then integrated to the
    # longest window in the batch and masked back to its own, which costs the
    # longest window once instead of every window in turn -- the batch axis is
    # very nearly free (kernels.py), the step count is not.
    horizons = np.broadcast_to(np.asarray(horizon, dtype=float), (batch,))
    lane_steps = np.array([round(h / (LYAPUNOV_EXP * DT)) for h in horizons])
    steps = int(lane_steps.max())
    uniform = bool((lane_steps == lane_steps[0]).all())

    # the penalty is meant to price the control we actually deploy, so it is
    # measured over the evaluation horizon rather than the training window --
    # a law that looks cheap over one Lyapunov time can cost several times
    # more once the run continues past it. None keeps the two windows equal.
    if effort_horizon is None:
        # the penalty rides the training trajectory, so it is differentiated
        # through the states like the task term is
        effort_steps = None
        pathwise = np.ones(batch, dtype=bool)
    else:
        effort_steps = round(effort_horizon / (LYAPUNOV_EXP * DT))
        pathwise = lane_steps == effort_steps

    # the two penalties are not the same objective -- one lets the gradient run
    # back through the states, the other freezes them -- so a batch may not mix
    # them. sweep.py keys its groups on this, so a grid never asks for it
    if pathwise.any() and not pathwise.all():
        raise ValueError(
            "this batch mixes lanes whose effort window equals their training "
            "window with lanes whose does not. Those differentiate the penalty "
            "differently (pathwise states vs frozen moments) and cannot share "
            "a run; split them, or set effort_horizon clear of every window"
        )

    use_moments = penalize_effort and not pathwise.all()

    if uniform:
        task_mask = effort_mask = None
    else:
        ticks = torch.arange(steps + 1, device=device).unsqueeze(-1)
        lanes = torch.as_tensor(lane_steps, device=device).unsqueeze(0)

        # task_loss averages over states 0..steps inclusive, effort_penalty
        # over the controls at 0..steps-1, so the two windows differ by one
        task_mask = ticks <= lanes
        effort_mask = ticks[:-1] < lanes

    lam = _as_batch(effort_weight, batch, device, dtype)
    lr = _as_batch(learning_rate, batch, device, dtype)

    if init is None:
        w, b = init_policy_params(batch, device=device, dtype=dtype)
    else:
        w, b = (
            torch.as_tensor(p, device=device, dtype=dtype).clone().requires_grad_()
            for p in init
        )
    state0 = (
        torch.as_tensor(np.asarray(state0, dtype=float), device=device, dtype=dtype)
        .expand(batch, 3)
        .contiguous()
    )

    opt = BatchedAdam([w, b], [lr.unsqueeze(-1), lr])
    history = torch.zeros(iters, batch, 3, device=device, dtype=dtype)
    grad_norm = torch.zeros(iters, batch, 3, device=device, dtype=dtype)
    # recorded whether or not it is asked for, so the captured graph is the
    # same either way; four numbers a lane is nothing next to the rollout
    params_seen = torch.zeros(iters, batch, 4, device=device, dtype=dtype)
    step_index = torch.zeros(1, dtype=torch.long, device=device)

    def iteration():
        traj = rollout_torch(
            state0,
            lambda s: linear_policy((w, b), s),
            steps=steps,
            integrator=integrator,
        )
        task = task_loss(traj, mask=task_mask)

        if use_moments:
            # moments of a rollout we do not differentiate through: over this
            # many Lyapunov times the pathwise gradient is pure amplified noise
            # (run -lgh to see it blow up), so the gradient reaches w and b
            # through the policy alone. Cheap, and well conditioned.
            effort = effort_from_moments(
                effort_moments(state0, (w, b), effort_steps, integrator), (w, b)
            )
        else:
            # unpenalized runs only log the number, so keep the rollout short
            effort = effort_penalty(traj, (w, b), mask=effort_mask)

        penalty = lam * effort
        loss = task + penalty if penalize_effort else task

        def norm(grad_w, grad_b):
            return (grad_w.square().sum(-1) + grad_b.square()).sqrt()

        # the policies are independent, so the sum hands each element exactly
        # the gradient it would have got from its own backward pass. The two
        # terms go back separately so each one's norm can be recorded; the
        # pathwise penalty shares the rollout, so that graph has to survive
        task.sum().backward(retain_graph=penalize_effort and not use_moments)
        task_norm = norm(w.grad, b.grad)

        if penalize_effort:
            penalty_w, penalty_b = torch.autograd.grad(penalty.sum(), (w, b))
            penalty_norm = norm(penalty_w, penalty_b)

            # added in place rather than assigned, so .grad keeps its identity
            # for the graph
            w.grad.add_(penalty_w)
            b.grad.add_(penalty_b)
        else:
            penalty_norm = torch.zeros_like(task_norm)

        # recorded in place, like history, so the graphed path never syncs
        grad_norm.index_copy_(
            0,
            step_index,
            torch.stack((task_norm, penalty_norm, norm(w.grad, b.grad)), dim=-1)
            .detach()
            .unsqueeze(0),
        )
        params_seen.index_copy_(
            0, step_index, torch.cat((w, b.unsqueeze(-1)), -1).detach().unsqueeze(0)
        )

        opt.step(step_index + 1)

        history.index_copy_(
            0,
            step_index,
            torch.stack((task, penalty, loss), dim=-1).detach().unsqueeze(0),
        )
        step_index.add_(1)
        opt.zero_grad()

    if use_graph and device == "cuda":
        _replay_graphed(iteration, iters, [w, b], opt, step_index)
    else:
        for _ in range(iters):
            iteration()

    recorded = (w.detach(), b.detach(), history.cpu().numpy(), grad_norm.cpu().numpy())

    return (*recorded, params_seen.cpu().numpy()) if record_params else recorded


def _replay_graphed(iteration, iters, params, opt, step_index, warmup=3):
    """Capture one training iteration and replay it, to stop paying for launches.

    A 221-step window is ~2000 kernel launches forward and back; at a few
    microseconds each that overhead *is* the iteration, and it does not shrink
    as the batch grows. Capturing the whole iteration once and replaying it
    drops it from ~35 ms to ~8 ms, flat out to at least B=4096.

    The warmup runs are real training steps -- they have to be, to force the
    Triton compile, the autograd graph and the optimizer state into existence
    before capture -- so everything they touched is reset before the capture,
    the parameters back to the random draw they started from.
    """
    initial = [p.detach().clone() for p in params]

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream):
        for _ in range(warmup):
            iteration()

    torch.cuda.current_stream().wait_stream(stream)

    with torch.no_grad():
        for p, p0 in zip(params, initial):
            p.copy_(p0)

    opt.reset()
    step_index.zero_()

    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        iteration()

    for _ in range(iters):
        graph.replay()

    torch.cuda.synchronize()


class RecedingRun(NamedTuple):
    """What a receding-horizon run hands back.

    The controller is the sequence of laws, one per window, so `w` and `b` carry
    a window axis in front of the batch axis. `traj` and `u` are the trajectory
    stitched from those windows -- law k applied over window k -- and are what a
    run is scored and drawn on. `history` and `grad_norm` are every window's
    training records end to end. A lane whose windows ran out before the
    batch's did (a longer window in a sweep) is padded past `windows[i]` with
    its last law frozen; `lane(i)` cuts that padding off.
    """

    w: np.ndarray  # (N, B, 3)
    b: np.ndarray  # (N, B)
    traj: np.ndarray  # (steps+1, B, 3)
    u: np.ndarray  # (steps+1, B)
    history: np.ndarray  # (N·iters, B, 3)
    grad_norm: np.ndarray  # (N·iters, B, 3)
    starts: np.ndarray  # (N, B, 3), where each window began
    windows: np.ndarray  # (B,), how many windows each lane actually ran
    params_seen: Optional[np.ndarray]  # (N·iters, B, 4), with record_params

    def lane(self, i):
        """Lane i alone, its padding dropped: every array loses the batch axis."""
        n = int(self.windows[i])
        rows = n * len(self.history) // len(self.w)

        return RecedingRun(
            w=self.w[:n, i],
            b=self.b[:n, i],
            traj=self.traj[:, i],
            u=self.u[:, i],
            history=self.history[:rows, i],
            grad_norm=self.grad_norm[:rows, i],
            starts=self.starts[:n, i],
            windows=n,
            params_seen=None if self.params_seen is None else self.params_seen[:rows, i],
        )


def train_receding_horizon(
    state0=(0, 1, 1.05),
    batch=1,
    window=1.0,
    total_horizon=10.0,
    iters=600,
    learning_rate=0.05,
    effort_weight=DEFAULT_EFFORT_WEIGHT,
    penalize_effort=True,
    integrator=rk4_step,
    device=None,
    dtype=None,
    use_graph=True,
    init=None,
    record_params=False,
    verbose=False,
):
    """Receding-horizon control: a law per window, each trained from where the last ended.

    `total_horizon` is split into back-to-back windows of `window` Lyapunov
    times, the last one shorter if the two don't divide. Window k trains its own
    (w, b) over that window alone -- task and effort both, pathwise -- starting
    from the state window k-1 ended in and from window k-1's law. Only the
    first window starts from a random law, and Adam starts afresh every window.
    The trained law is then applied over its window to find where the next one
    starts: rolled out with the law as it finished, not as it stood at the last
    gradient, which is one Adam step behind.

    `state0`, `learning_rate`, `effort_weight` and `window` may each be per
    lane, as in train_policy_batched. A lane with a longer window runs out of
    windows first; it then sits out the rest with a learning rate of zero,
    which leaves its law exactly where it was.
    """
    unit = LYAPUNOV_EXP * DT
    windows_lt = np.broadcast_to(np.asarray(window, dtype=float), (batch,))
    # in steps rather than Lyapunov times, so the windows tile the horizon
    # exactly instead of drifting by a rounding error each
    seg = np.array([round(h / unit) for h in windows_lt])
    total = round(total_horizon / unit)

    if (seg < 1).any() or total < 1:
        raise ValueError("the window and the total horizon must each be a step or more")

    windows = -(-total // seg)
    n = int(windows.max())
    lr = np.broadcast_to(np.asarray(learning_rate, dtype=float), (batch,))

    current = np.broadcast_to(np.asarray(state0, dtype=float), (batch, 3)).copy()
    traj = np.empty((total + 1, batch, 3))
    u = np.empty((total + 1, batch))
    traj[0] = current

    laws_w = np.empty((n, batch, 3))
    laws_b = np.empty((n, batch))
    starts = np.empty((n, batch, 3))
    histories, norms, seen = [], [], []

    for k in range(n):
        offset = k * seg
        live = k < windows
        # a finished lane still needs some window to sit in; its own does
        steps = np.where(live, np.minimum(seg, total - offset), seg)
        starts[k] = current

        w, b, history, grad_norm, *params_seen = train_policy_batched(
            state0=current,
            batch=batch,
            horizon=steps * unit,
            effort_horizon=None,
            iters=iters,
            learning_rate=np.where(live, lr, 0.0),
            effort_weight=effort_weight,
            penalize_effort=penalize_effort,
            integrator=integrator,
            device=device,
            dtype=dtype,
            use_graph=use_graph,
            init=init,
            record_params=record_params,
        )

        w = w.cpu().double().numpy()
        b = b.cpu().double().numpy()
        laws_w[k], laws_b[k] = w, b
        init = (w, b)

        histories.append(history)
        norms.append(grad_norm)
        seen.extend(params_seen)

        # apply the law over its window. float64 on the cpu like every other
        # rollout that is scored, whatever precision the training ran in
        piece, controls = rollout_numpy_batched(
            current, w, b, steps=int(steps[live].max()), integrator=integrator
        )

        for i in np.flatnonzero(live):
            start, length = offset[i], steps[i]

            traj[start + 1 : start + length + 1, i] = piece[1 : length + 1, i]
            # the control at a window's first state is that window's law's:
            # it is the one acting from there
            u[start : start + length, i] = controls[:length, i]
            current[i] = piece[length, i]

        if verbose:
            _log_window(k, n, starts[k], history, w, b, live)

    # the last state's control is never applied, but the figures plot it
    last = windows - 1
    lanes = np.arange(batch)
    u[total] = (traj[total] * laws_w[last, lanes]).sum(-1) + laws_b[last, lanes]

    return RecedingRun(
        w=laws_w,
        b=laws_b,
        traj=traj,
        u=u,
        history=np.concatenate(histories),
        grad_norm=np.concatenate(norms),
        starts=starts,
        windows=windows,
        params_seen=np.concatenate(seen) if record_params else None,
    )


def _log_window(k, n, starts, history, w, b, live):
    # read off the recorded history once the window is done, which on the
    # graphed path is the only way that doesn't sync the device per iteration
    first, last = history[0, :, 2], history[-1, :, 2]

    if len(live) == 1:
        x, y, z = starts[0]
        print(
            f"window {k + 1:3d}/{n}  start ({x:+7.2f}, {y:+7.2f}, {z:+6.2f})  "
            f"loss {first[0]:.4f} -> {last[0]:.4f}   "
            f"w {w[0].round(3)}  b {b[0]:+.3f}",
            flush=True,
        )
    else:
        print(
            f"window {k + 1:3d}/{n}  {live.sum()} lane(s) live  "
            f"mean loss {first[live].mean():.4f} -> {last[live].mean():.4f}",
            flush=True,
        )


def train_policy(
    state0=(0, 1, 1.05),
    horizon=1.0,
    total_horizon=10.0,
    iters=600,
    lr=0.1,
    effort_weight=DEFAULT_EFFORT_WEIGHT,
    penalize_effort=True,
    integrator=rk4_step,
    record_params=False,
    verbose=True,
    device=None,
    dtype=None,
    use_graph=True,
):
    """One receding-horizon run, as a batch of one: its RecedingRun, batch axis dropped."""
    return train_receding_horizon(
        state0=state0,
        batch=1,
        window=horizon,
        total_horizon=total_horizon,
        iters=iters,
        learning_rate=lr,
        effort_weight=effort_weight,
        penalize_effort=penalize_effort,
        integrator=integrator,
        device=device,
        dtype=dtype,
        use_graph=use_graph,
        record_params=record_params,
        verbose=verbose,
    ).lane(0)


if __name__ == "__main__":
    run = train_policy(lr=0.02)
    print(f"success over the stitched trajectory: {success_fraction(run.traj):.4f}")
