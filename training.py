"""The learned controller: policy, objectives, and the training loop.

The policy is a linear feedback law u = w·s + b, trained by differentiating
straight through the integrator, so the training window is limited by how far
the gradient survives the attractor's exponential stretching.

Everything here carries an optional leading batch axis: w is (B, 3), b is (B,),
and the objectives reduce over time only, returning one number per policy. B
independent policies then train in one set of kernels instead of B sets, which
is what makes a sweep worth putting on a GPU at all -- see kernels.py for why
a single 3-component state emphatically is not.
"""

import numpy as np
import torch

from kernels import default_dtype, moment_stats, resolve_device
from lorenz import DT, LYAPUNOV_EXP, euler_step, rollout_torch
from params import DEFAULT_EFFORT_WEIGHT

SOFTNESS = 2.0
U_REF = 60.0


def init_policy_params(batch=None, device="cpu", dtype=torch.float64):
    shape = () if batch is None else (batch,)

    w = torch.zeros((*shape, 3), dtype=dtype, device=device, requires_grad=True)
    b = torch.zeros(shape, dtype=dtype, device=device, requires_grad=True)

    return w, b


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


def effort_moments(state0, params, steps, integrator=euler_step):
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
    penalize_effort=False,
    integrator=euler_step,
    device=None,
    dtype=None,
    use_graph=True,
):
    """Train `batch` independent policies at once.

    `learning_rate`, `effort_weight` and `horizon` may each be a scalar or a
    length-`batch` sequence; those are the axes a grid can vary within one
    run. The first two are per-element numbers the arithmetic carries anyway.
    `horizon` is the interesting one: it does change the step count, so the
    batch integrates to the longest window in it and each lane's loss is
    masked back to its own -- which costs max(steps) rather than sum(steps).
    The integrator, the iteration count and the evaluation window still have
    to match across a batch, so those stay separate calls.

    Returns (w, b, history), history being (iters, batch, 3) of
    (task, λ·effort, total) -- the same three series the scalar loop recorded.
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

    w, b = init_policy_params(batch, device=device, dtype=dtype)
    state0 = (
        torch.as_tensor(np.asarray(state0, dtype=float), device=device, dtype=dtype)
        .expand(batch, 3)
        .contiguous()
    )

    opt = BatchedAdam([w, b], [lr.unsqueeze(-1), lr])
    history = torch.zeros(iters, batch, 3, device=device, dtype=dtype)
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

        # the policies are independent, so the sum hands each element exactly
        # the gradient it would have got from its own backward pass
        loss.sum().backward()

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

    return w.detach(), b.detach(), history.cpu().numpy()


def _replay_graphed(iteration, iters, params, opt, step_index, warmup=3):
    """Capture one training iteration and replay it, to stop paying for launches.

    A 221-step window is ~2000 kernel launches forward and back; at a few
    microseconds each that overhead *is* the iteration, and it does not shrink
    as the batch grows. Capturing the whole iteration once and replaying it
    drops it from ~35 ms to ~8 ms, flat out to at least B=4096.

    The warmup runs are real training steps -- they have to be, to force the
    Triton compile, the autograd graph and the optimizer state into existence
    before capture -- so everything they touched is reset before the capture.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream):
        for _ in range(warmup):
            iteration()

    torch.cuda.current_stream().wait_stream(stream)

    with torch.no_grad():
        for p in params:
            p.zero_()

    opt.reset()
    step_index.zero_()

    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        iteration()

    for _ in range(iters):
        graph.replay()

    torch.cuda.synchronize()


def train_policy(
    state0=[0, 1, 1.05],
    params=None,
    horizon=1.0,
    effort_horizon=None,
    iters=600,
    lr=0.1,
    effort_weight=DEFAULT_EFFORT_WEIGHT,
    penalize_effort=False,
    integrator=euler_step,
    history=None,
    verbose=True,
    device=None,
    dtype=None,
    use_graph=True,
):
    """One policy, as a batch of one. `history`, if given, collects the triples."""
    w, b, recorded = train_policy_batched(
        state0=state0,
        batch=1,
        horizon=horizon,
        effort_horizon=effort_horizon,
        iters=iters,
        learning_rate=lr,
        effort_weight=effort_weight,
        penalize_effort=penalize_effort,
        integrator=integrator,
        device=device,
        dtype=dtype,
        use_graph=use_graph,
    )

    if history is not None:
        history.extend(tuple(row) for row in recorded[:, 0, :])

    if verbose:
        # read back off the recorded history rather than printing inside the
        # loop, which on the graphed path would sync the device every iteration
        for i in range(0, iters, 20):
            task, penalty, total = recorded[i, 0]
            print(
                f"iter {i:4d}  loss {total:.4f}   task {task:.4f}   pen {penalty:.4f}"
            )

        print(f"w {w[0].cpu().numpy().round(3)}   b {b[0].item():+.3f}")

    # the rest of the project -- figures.py above all -- works in float64 on
    # the cpu, so the single-run door hands back what it always did whatever
    # device and precision the batch actually trained in
    return w[0].cpu().to(torch.float64), b[0].cpu().to(torch.float64)


if __name__ == "__main__":
    w, b = train_policy(lr=0.02)
    print("learned feedback law: w =", w.cpu().numpy(), ". (x,y,z) +", b.item())
