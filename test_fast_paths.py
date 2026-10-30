"""What has to stay true for the batched and CUDA paths to be trustworthy.

    python test_fast_paths.py

The point of these is narrow: the fast paths must compute *the same thing* as
the plain ones. They are all equivalences, and where a comparison has to hold
exactly, it is asserted exactly.

The one thing they deliberately do not check is a long rollout against a long
rollout. Over the evaluation window this system amplifies a last-bit difference
into a several-percent one -- `test_chaos_dominates_precision` measures that
directly -- so agreement there would be a statement about the attractor, not
about the code. Short windows are where a bug has nowhere to hide.
"""

import numpy as np
import torch

from kernels import cuda_available, moment_stats_cuda, moment_stats_numpy
from lorenz import (
    DT,
    LYAPUNOV_EXP,
    euler_step,
    limit_control,
    lorenz_rhs,
    random_attractor_state,
    rk4_step,
    rollout_numpy,
    rollout_numpy_batched,
)
from params import resolve_start
from training import (
    BatchedAdam,
    effort_moments,
    init_policy_params,
    linear_policy,
    train_policy_batched,
    train_receding_horizon,
    train_receding_prefixes,
)

CUDA = cuda_available()


def test_rhs_is_batch_polymorphic():
    """A batch of states must step exactly as each state would alone."""
    rng = np.random.default_rng(0)
    states = rng.normal(0, 10, (8, 3))
    us = rng.normal(0, 5, 8)

    batched = lorenz_rhs(states, us)

    for i, (state, u) in enumerate(zip(states, us)):
        assert np.array_equal(lorenz_rhs(state, u), batched[i])


def test_callable_control_is_honoured_on_every_row():
    """A rollout must apply `u` through `eval_control` everywhere, last row included.

    `derivs[steps]` is filled by its own call outside the integrator loop, so it
    is the one row that can quietly skip `eval_control` while every other row
    still looks right. Two ways in: a state-independent callable has to give
    back exactly what the bare scalar gives, and a real policy has to leave the
    closed-loop field -- not the uncontrolled one -- in that final row.
    """
    state0 = [0.0, 1.0, 1.05]

    traj, derivs = rollout_numpy(state0, steps=50, u=2.5)
    traj_fn, derivs_fn = rollout_numpy(state0, steps=50, u=lambda state: 2.5)

    assert np.array_equal(traj, traj_fn)
    assert np.array_equal(derivs, derivs_fn)

    w, b = np.array([-2.7, 0.4, -0.1]), 22.9
    policy = lambda state: linear_policy((w, b), state)  # noqa: E731
    traj, derivs = rollout_numpy(state0, steps=50, u=policy)

    final = lorenz_rhs(traj[-1], policy(traj[-1]))

    assert np.array_equal(derivs[-1], final)
    assert not np.array_equal(derivs[-1], lorenz_rhs(traj[-1], 0.0))


def test_batched_adam_matches_torch():
    """Per-element learning rates are the only intended departure from Adam."""
    torch.manual_seed(0)
    n = 5
    reference = [
        torch.zeros(n, 3, dtype=torch.float64, requires_grad=True),
        torch.zeros(n, dtype=torch.float64, requires_grad=True),
    ]
    mine = [p.detach().clone().requires_grad_(True) for p in reference]

    lr = torch.full((n,), 0.05, dtype=torch.float64)
    torch_adam = torch.optim.Adam(reference, lr=0.05)
    batched = BatchedAdam(mine, [lr.unsqueeze(-1), lr])
    step = torch.zeros(1, dtype=torch.long)

    for _ in range(200):
        grads = [
            torch.randn(n, 3, dtype=torch.float64),
            torch.randn(n, dtype=torch.float64),
        ]

        for params in (reference, mine):
            for p, g in zip(params, grads):
                p.grad = g.clone()

        torch_adam.step()
        batched.step(step + 1)
        step.add_(1)

    for ref, got in zip(reference, mine):
        assert (ref - got).abs().max() < 1e-14, (ref - got).abs().max()


def test_batched_adam_learning_rates_are_independent():
    lrs = torch.tensor([0.01, 0.02, 0.05, 0.1, 0.2], dtype=torch.float64)
    p = [torch.zeros(5, 3, dtype=torch.float64, requires_grad=True)]
    opt = BatchedAdam(p, [lrs.unsqueeze(-1)])
    step = torch.zeros(1, dtype=torch.long)

    # a constant unit gradient makes Adam's step exactly the learning rate
    for _ in range(10):
        p[0].grad = torch.ones(5, 3, dtype=torch.float64)
        opt.step(step + 1)
        step.add_(1)

    assert torch.allclose(p[0][:, 0], -10 * lrs, atol=1e-6)


def test_numpy_moments_match_the_scalar_loop():
    """The batched accumulation is the same accumulation, one point at a time."""
    rng = np.random.default_rng(1)
    w = rng.normal(0, 0.5, (6, 3))
    b = rng.normal(0, 1, 6)
    state0 = np.broadcast_to(np.array([0.0, 1.0, 1.05]), (6, 3)).copy()

    M, m = moment_stats_numpy(state0, w, b, 300)

    for i in range(6):
        params = (torch.tensor(w[i]), torch.tensor(b[i]))
        M_one, m_one = effort_moments([0, 1, 1.05], params, 300)

        assert np.allclose(M[i], M_one, rtol=1e-12)
        assert np.allclose(m[i], m_one, rtol=1e-12)


def test_triton_kernel_matches_numpy():
    """Both integrators, over a window short enough that chaos is not the story."""
    if not CUDA:
        print("  (skipped: no CUDA device)")
        return

    rng = np.random.default_rng(2)
    w = rng.normal(0, 0.5, (37, 3))
    b = rng.normal(0, 1, 37)
    state0 = np.broadcast_to(np.array([0.0, 1.0, 1.05]), (37, 3)).copy()

    for integrator in (euler_step, rk4_step):
        M, m = moment_stats_numpy(state0, w, b, 200, integrator)
        kernel = moment_stats_cuda(
            torch.tensor(state0, device="cuda"),
            torch.tensor(w, device="cuda"),
            torch.tensor(b, device="cuda"),
            200,
            integrator,
        )
        M_cuda, m_cuda = (t.cpu().numpy() for t in kernel)

        assert np.abs(M_cuda - M).max() / np.abs(M).max() < 1e-13
        assert np.abs(m_cuda - m).max() / np.abs(m).max() < 1e-13


def test_batching_does_not_change_the_answer():
    """B policies trained together must equal B trained alone.

    Held to the short window, where the objective carries no long rollout, so
    any difference would be the batching itself rather than the attractor.
    Most elements come back bit-identical; the tolerance is there for the ones
    where batching reassociates a reduction, which is worth a last bit or two
    of float64 and nothing more.
    """
    if not CUDA:
        print("  (skipped: no CUDA device)")
        return

    lrs = [0.02, 0.05, 0.08, 0.05]
    lams = [0.05, 0.10, 0.15, 0.10]
    common = dict(
        horizon=1.0,
        iters=25,
        penalize_effort=True,
        device="cuda",
        dtype=torch.float64,
    )

    # each lane's random start is fixed up front so its solo run can share it;
    # lane 3 repeats lane 1 in full, starting law included
    w0, b0 = (p.detach() for p in init_policy_params(4))
    w0[3], b0[3] = w0[1], b0[1]

    w, b, history, grad_norm = train_policy_batched(
        batch=4,
        effort_weight=lams,
        learning_rate=lrs,
        use_graph=True,
        init=(w0, b0),
        **common,
    )

    for i, (lr, lam) in enumerate(zip(lrs, lams)):
        # eager is ~4x slower than a replay, so one eager lane is enough to
        # hold the graph to the eager arithmetic; the rest replay too
        w_one, b_one, history_one, grad_norm_one = train_policy_batched(
            batch=1,
            effort_weight=[lam],
            learning_rate=[lr],
            use_graph=i > 0,
            init=(w0[i : i + 1], b0[i : i + 1]),
            **common,
        )

        assert (w[i] - w_one[0]).abs().max() < 1e-12, (i, w[i], w_one[0])
        assert (b[i] - b_one[0]).abs().max() < 1e-12, (i, b[i], b_one[0])
        assert np.abs(history[:, i] - history_one[:, 0]).max() < 1e-15
        # the graphed lane must record the same norms its solo run did
        assert np.allclose(grad_norm[:, i], grad_norm_one[:, 0], rtol=1e-12, atol=0)

    # two lanes given identical settings do the identical arithmetic, so these
    # have no reassociation to excuse a difference and must match exactly
    assert torch.equal(w[1], w[3])
    assert torch.equal(b[1], b[3])


def test_a_run_shorter_than_the_warmup_is_graphed():
    """The graph's warmup steps are real ones, but they must not outrun the records.

    Those are sized to `iters`, so a warmup that walked down them would write
    past a run shorter than itself -- a device-side assert rather than a result.
    """
    if not CUDA:
        print("  (skipped: no CUDA device)")
        return

    w0, b0 = (p.detach() for p in init_policy_params(1))
    common = dict(horizon=0.25, device="cuda", dtype=torch.float64, init=(w0, b0))

    for iters in (0, 1, 2):
        graphed = train_policy_batched(iters=iters, use_graph=True, **common)
        eager = train_policy_batched(iters=iters, use_graph=False, **common)

        assert (graphed[0] - eager[0]).abs().max() < 1e-12, (iters, graphed[0], eager[0])
        assert graphed[2].shape == eager[2].shape == (iters, 1, 3)
        assert np.allclose(graphed[2], eager[2], rtol=0, atol=1e-15), iters


def test_mixed_windows_match_their_own_runs():
    """A short lane riding along a longer batch must equal a run at its length.

    Every lane here integrates to the longest window in the batch and is masked
    back to its own, so this is the check that the mask covers exactly the
    steps a standalone run would have taken -- one step out in either direction
    and the shorter lanes would drift.
    """
    if not CUDA:
        print("  (skipped: no CUDA device)")
        return

    horizons = [1.0, 1.25, 1.5, 1.75, 2.0]
    lams = [0.06, 0.08, 0.10, 0.12, 0.14]
    common = dict(
        iters=25,
        penalize_effort=True,
        learning_rate=0.05,
        device="cuda",
        dtype=torch.float64,
    )

    w0, b0 = (p.detach() for p in init_policy_params(5))

    w, b, _, _ = train_policy_batched(
        batch=5,
        horizon=horizons,
        effort_weight=lams,
        use_graph=True,
        init=(w0, b0),
        **common,
    )

    for i, (horizon, lam) in enumerate(zip(horizons, lams)):
        w_one, b_one, _, _ = train_policy_batched(
            batch=1,
            horizon=horizon,
            effort_weight=[lam],
            # one eager lane, as in the batching test
            use_graph=i > 0,
            init=(w0[i : i + 1], b0[i : i + 1]),
            **common,
        )

        assert (w[i] - w_one[0]).abs().max() < 1e-12, (horizon, w[i], w_one[0])
        assert (b[i] - b_one[0]).abs().max() < 1e-12, (horizon, b[i], b_one[0])


def test_mixed_starts_match_their_own_runs():
    """A batch of initial conditions must equal a run from each start alone.

    state0 is the newest axis a grid can batch over, and it is the one where a
    bug would be quietest: a batch that silently shared one start would still
    train, still converge and still plot -- it would just be answering a
    question nobody asked. So this checks the laws exactly, and checks that
    each lane's rollout actually began where it was told to.
    """
    starts = [(0.0, 1.0, 1.05), (0.5, 1.0, 1.05), (1.0, 1.0, 1.05), (-0.3, 0.2, 2.0)]
    common = dict(
        horizon=1.0,
        iters=25,
        penalize_effort=True,
        learning_rate=0.05,
        effort_weight=0.07,
        device="cpu",
        dtype=torch.float64,
        use_graph=False,
    )

    w0, b0 = (p.detach() for p in init_policy_params(len(starts)))

    w, b, _, grad_norm = train_policy_batched(
        state0=starts, batch=len(starts), init=(w0, b0), **common
    )

    for i, start in enumerate(starts):
        w_one, b_one, _, grad_norm_one = train_policy_batched(
            state0=[start], batch=1, init=(w0[i : i + 1], b0[i : i + 1]), **common
        )

        assert (w[i] - w_one[0]).abs().max() < 1e-12, (start, w[i], w_one[0])
        assert (b[i] - b_one[0]).abs().max() < 1e-12, (start, b[i], b_one[0])
        # summing the lanes' losses must not leak one lane's gradient into another
        assert np.allclose(grad_norm[:, i], grad_norm_one[:, 0], rtol=1e-12, atol=0)

    # the total is the norm of the sum of the other two, so it has to sit
    # between their difference and their sum -- a swapped column would not
    task_norm, penalty_norm, total_norm = np.moveaxis(grad_norm, -1, 0)
    slack = 1e-12 * (task_norm + penalty_norm)
    assert (total_norm <= task_norm + penalty_norm + slack).all()
    assert (total_norm >= np.abs(task_norm - penalty_norm) - slack).all()
    assert (penalty_norm[1:] > 0).all()

    # the evaluation rollout broadcasts state0 against w rather than expanding
    # it, so it is a separate place the starts could collapse
    traj, _ = rollout_numpy_batched(starts, w.numpy(), b.numpy(), steps=4)

    assert np.array_equal(traj[0], np.asarray(starts, dtype=float)), traj[0]


def test_chaos_dominates_precision():
    """Why float32 on the GPU is not the thing costing accuracy.

    Over the evaluation window the moments move further under a nudge of ~1e-12
    to the initial condition, in float64 throughout, than they do between
    float32 and float64. Any argument for fp64 here has to clear that bar first.

    The bar is read off several nudges rather than one. What a single nudge
    costs is itself a few-percent number drawn from a wide spread -- these four
    span a factor of five -- so one sample of it is not a bar, it is a coin
    flip that happens to have landed above fp32.
    """
    params = (torch.zeros(3, dtype=torch.float64), torch.zeros((), dtype=torch.float64))
    steps = 22085

    # effort_moments hands back whatever kind its parameters were, so pull the
    # comparison into numpy rather than subtracting a tensor from an array
    M = effort_moments([0, 1, 1.05], params, steps)[0].numpy()
    nudged = [
        np.abs(effort_moments([0, 1 + eps, 1.05], params, steps)[0].numpy() - M).max()
        / np.abs(M).max()
        for eps in (1e-12, 2e-12, 5e-12, -1e-12)
    ]

    chaos = float(np.median(nudged))
    assert chaos > 1e-2, f"expected the attractor to dominate, got {chaos:.2e}"

    if CUDA:
        state0 = torch.tensor([[0.0, 1.0, 1.05]], device="cuda", dtype=torch.float32)
        M32, _ = moment_stats_cuda(
            state0,
            torch.zeros(1, 3, device="cuda", dtype=torch.float32),
            torch.zeros(1, device="cuda", dtype=torch.float32),
            steps,
        )
        precision = np.abs(M32[0].cpu().numpy() - M).max() / np.abs(M).max()

        assert precision < chaos, f"fp32 {precision:.2e} vs 1e-12 nudges {chaos:.2e}"
        print(f"  fp32 costs {precision:.1%}; a 1e-12 nudge costs {chaos:.1%}")


# a batching or hand-off bug shows from the first step, so a few iterations
# per window are as sharp a check as many
RECEDING = dict(
    iters=10,
    learning_rate=0.05,
    effort_weight=0.07,
    penalize_effort=True,
    device="cpu",
    dtype=torch.float64,
    use_graph=False,
)


def test_one_window_is_a_plain_run():
    """With the window as long as the horizon, RHC is exactly one ordinary run."""
    start = (0.0, 1.0, 1.05)
    w0, b0 = (p.detach() for p in init_policy_params(1))

    run = train_receding_horizon(
        state0=start, window=1.0, total_horizon=1.0, init=(w0, b0), **RECEDING
    )
    w, b, history, _ = train_policy_batched(
        state0=start, horizon=1.0, init=(w0, b0), **RECEDING
    )

    assert run.windows.tolist() == [1]
    assert np.array_equal(run.w[0], w.numpy())
    assert np.array_equal(run.b[0], b.numpy())
    assert np.array_equal(run.history, history)

    steps = round(1.0 / (LYAPUNOV_EXP * DT))
    traj, u = rollout_numpy_batched([start], w.numpy(), b.numpy(), steps=steps)

    assert np.array_equal(run.traj, traj)
    assert np.array_equal(run.u, u)


def test_windows_hand_off_state_and_law():
    """Window 2 must start where window 1's trained law left the system, from that law."""
    start = (0.0, 1.0, 1.05)
    w0, b0 = (p.detach() for p in init_policy_params(1))
    seg = round(1.0 / (LYAPUNOV_EXP * DT))

    run = train_receding_horizon(
        state0=start, window=1.0, total_horizon=2.0, init=(w0, b0), **RECEDING
    )

    assert run.windows.tolist() == [2]
    assert np.array_equal(run.starts[1], run.traj[seg])

    w, b, _, _ = train_policy_batched(
        state0=run.traj[seg],
        horizon=1.0,
        init=(run.w[0], run.b[0]),
        **RECEDING,
    )

    assert np.array_equal(run.w[1], w.numpy())
    assert np.array_equal(run.b[1], b.numpy())

    # the stitched trajectory is law 1 over window 1, then law 2 from there
    first, _ = rollout_numpy_batched([start], run.w[0], run.b[0], steps=seg)
    second, u2 = rollout_numpy_batched(run.traj[seg], run.w[1], run.b[1], steps=seg)

    assert np.array_equal(run.traj[: seg + 1], first)
    assert np.array_equal(run.traj[seg:], second)
    assert np.array_equal(run.u[seg:], u2)


def test_control_is_held_over_each_window():
    """u is sampled once at each window's start and held until the next one."""
    seg = round(1.0 / (LYAPUNOV_EXP * DT))
    # ungated and uncapped, so every window holds its law's own sample
    run = train_receding_horizon(
        window=1.0, total_horizon=2.0, x_threshold=np.inf, u_max=np.inf, **RECEDING
    )

    for k in range(2):
        held = (run.starts[k, 0] * run.w[k, 0]).sum(-1) + run.b[k, 0]
        window = run.u[k * seg : (k + 1) * seg, 0]

        assert np.array_equal(window, np.full(seg, held)), (k, window[:3], held)

    # the two windows' values really are different samples
    assert run.u[seg - 1, 0] != run.u[seg, 0]


def test_a_remainder_window_is_shorter():
    """A horizon the window doesn't divide ends on a short window, not an overshoot."""
    run = train_receding_horizon(window=1.0, total_horizon=2.5, **{**RECEDING, "iters": 2})

    seg = round(1.0 / (LYAPUNOV_EXP * DT))

    assert run.windows.tolist() == [3]
    # two and a half windows, counted in windows rather than in raw steps
    assert len(run.traj) == round(2.5 * seg) + 1
    assert run.steps.tolist() == [round(2.5 * seg)]
    assert np.isfinite(run.traj).all() and np.isfinite(run.u).all()


def test_finished_lanes_stay_frozen():
    """Lanes with different windows batched together must each equal their solo run.

    The longer-window lane runs out of windows first and sits out the rest at a
    learning rate of zero, so its law, its trajectory and its records all have
    to come back as if it had trained alone.
    """
    windows = [1.0, 2.0]
    w0, b0 = (p.detach() for p in init_policy_params(2))

    run = train_receding_horizon(
        batch=2, window=windows, total_horizon=2.0, init=(w0, b0), **RECEDING
    )

    assert run.windows.tolist() == [2, 1]

    for i, window in enumerate(windows):
        one = train_receding_horizon(
            window=window,
            total_horizon=2.0,
            init=(w0[i : i + 1], b0[i : i + 1]),
            **RECEDING,
        ).lane(0)
        mine = run.lane(i)

        assert np.abs(mine.w - one.w).max() < 1e-12, (window, mine.w, one.w)
        assert np.abs(mine.traj - one.traj).max() < 1e-9, window
        assert mine.history.shape == one.history.shape

    # padding past a lane's last window repeats its last law exactly
    assert np.array_equal(run.w[1, 1], run.w[0, 1])


def test_random_start_is_reproducible():
    """A seed names one start, on the attractor; a given start is taken as it is."""
    a = random_attractor_state(np.random.default_rng(7))
    b = random_attractor_state(np.random.default_rng(7))
    c = random_attractor_state(np.random.default_rng(8))

    assert a == b and a != c
    x, y, z = a
    assert abs(x) < 25 and abs(y) < 30 and 0 < z < 55, a

    assert resolve_start({"initial_condition": None, "seed": 7}) == (a, 7)
    assert resolve_start({"initial_condition": (0, 1, 1.05), "seed": None}) == (
        (0.0, 1.0, 1.05),
        None,
    )

    drawn, seed = resolve_start({"initial_condition": None, "seed": None})
    assert resolve_start({"initial_condition": None, "seed": seed}) == (drawn, seed)

    try:
        resolve_start({"initial_condition": (0, 1, 1.05), "seed": 7})
    except SystemExit:
        return

    raise AssertionError("a given start and a seed together should have been refused")


def _same_run(read, alone, history_atol=0.0):
    """Two single-lane RecedingRuns that must agree to the last bit.

    `history_atol` loosens the recorded loss alone: from some starts the
    batched time mean rounds differently from the lone one in the last place,
    which the law, and so everything else, never sees.
    """
    assert read.windows == alone.windows, (read.windows, alone.windows)

    for field in ("w", "b", "traj", "u", "history", "grad_norm", "starts"):
        mine, theirs = getattr(read, field), getattr(alone, field)

        assert mine.shape == theirs.shape, (field, mine.shape, theirs.shape)

        if field == "history" and history_atol:
            assert np.allclose(mine, theirs, rtol=0, atol=history_atol), field
        else:
            assert np.array_equal(mine, theirs), field


def _alone(horizon, init, window=1.0):
    return train_receding_horizon(
        window=window, total_horizon=horizon, init=init, **RECEDING
    ).lane(0)


def test_prefix_equals_a_shorter_run():
    """A horizon at a window boundary is read off the longer run exactly."""
    init = tuple(p.detach() for p in init_policy_params(1))
    runs = train_receding_prefixes([3.0, 2.0, 1.0], init=init, **RECEDING)[0]

    _same_run(runs[0], _alone(3.0, init))
    _same_run(runs[1], _alone(2.0, init))
    _same_run(runs[2], _alone(1.0, init))

    # whole windows nest with no rounding left over, so this is 10 windows
    # read at 9, not 9 whole windows and a short tenth
    seg = round(1.0 / (LYAPUNOV_EXP * DT))
    assert runs[1].windows == 2 and len(runs[1].traj) == 2 * seg + 1


def test_remainder_branch_equals_a_shorter_run():
    """A horizon part-way through a window ends on its own shorter window."""
    init = tuple(p.detach() for p in init_policy_params(1))
    runs = train_receding_prefixes([3.0, 2.5], init=init, **RECEDING)[0]

    _same_run(runs[1], _alone(2.5, init))
    # and the branch leaves the long run it hangs off untouched
    _same_run(runs[0], _alone(3.0, init))


def test_prefix_shorter_than_a_window():
    """A horizon inside the first window is one short window from the first law."""
    init = tuple(p.detach() for p in init_policy_params(1))
    runs = train_receding_prefixes([2.0, 0.5], init=init, **RECEDING)[0]

    _same_run(runs[1], _alone(0.5, init))

    # the law it starts from is the one the long run recorded, whether or not
    # the caller handed one in
    torch.manual_seed(3)
    drawn = train_receding_prefixes([2.0, 0.5], **RECEDING)[0]
    _same_run(drawn[1], _alone(0.5, (drawn[0].init_w[None], drawn[0].init_b[None])))


# on the right lobe, well past the default threshold of 2, so the first window
# starting here is gated off
RIGHT_LOBE = (8.0, 8.0, 27.0)


def test_gate_zeroes_a_lane_past_the_threshold():
    """A start at x >= x_threshold holds u = 0 and runs uncontrolled; one below is untouched."""
    starts = np.array([RIGHT_LOBE, (-8.0, -8.0, 27.0)])
    w = np.array([[0.3, -0.2, 0.1], [0.3, -0.2, 0.1]])
    b = np.array([1.5, 1.5])
    steps = 50

    traj, u = rollout_numpy_batched(starts, w, b, steps=steps, x_threshold=2.0)
    free, _ = rollout_numpy(RIGHT_LOBE, steps=steps, u=0.0)
    ungated, u_ungated = rollout_numpy_batched(
        starts, w, b, steps=steps, x_threshold=np.inf
    )

    assert (u[:, 0] == 0).all()
    assert np.array_equal(traj[:, 0], free)
    assert np.array_equal(traj[:, 1], ungated[:, 1])
    assert np.array_equal(u[:, 1], u_ungated[:, 1]) and (u[:, 1] != 0).all()


def test_a_gated_window_leaves_its_law_untrained():
    """A window starting past the threshold gets no gradient and passes its law on."""
    seg = round(1.0 / (LYAPUNOV_EXP * DT))
    w0, b0 = (p.detach() for p in init_policy_params(1))

    run = train_receding_horizon(
        state0=RIGHT_LOBE, window=1.0, total_horizon=2.0, init=(w0, b0), **RECEDING
    ).lane(0)
    rows = RECEDING["iters"]
    free, _ = rollout_numpy(RIGHT_LOBE, steps=seg, u=0.0)

    assert (run.u[:seg] == 0).all()
    assert np.array_equal(run.traj[: seg + 1], free)
    assert np.array_equal(run.w[0], w0.numpy()[0])
    assert np.array_equal(run.b[0], b0.numpy()[0])
    # no task gradient, no effort gradient, and no effort to pay
    assert (run.grad_norm[:rows] == 0).all()
    assert (run.history[:rows, 1] == 0).all()


def _two_limits(horizons, name, values, state0=RIGHT_LOBE, init=None, **same):
    """Two lanes that differ only in one limit, batched, and each alone."""
    if init is None:
        init = tuple(p.detach() for p in init_policy_params(2))

    batched = train_receding_prefixes(
        horizons, state0=state0, batch=2, init=init, **{name: values}, **RECEDING
    )

    for i, value in enumerate(values):
        for run, horizon in zip(batched[i], horizons):
            alone = train_receding_horizon(
                state0=state0,
                window=1.0,
                total_horizon=horizon,
                init=(init[0][i : i + 1], init[1][i : i + 1]),
                **{name: value},
                **RECEDING,
            ).lane(0)

            _same_run(run, alone, **same)
            assert getattr(run, name) == value

    return batched


def test_thresholds_batch_per_lane():
    """Each lane is gated by its own threshold, as if it had trained alone."""
    batched = _two_limits([2.0], "x_threshold", [2.0, np.inf])

    # and the two really did differ: only the gated lane held zero at the start
    assert batched[0][0].u[0] == 0 and batched[1][0].u[0] != 0


def test_remainder_branches_keep_their_threshold():
    """A remainder window trained as a branch is gated by its own lane's threshold."""
    _two_limits([2.0, 1.5], "x_threshold", [2.0, np.inf])


# on the left lobe, below the default threshold, so every window starting here
# is controlled
LEFT_LOBE = (-8.0, -8.0, 27.0)


def test_cap_clips_the_held_control():
    """A law past ±u_max holds exactly ±u_max; one inside the cap is untouched."""
    starts = np.array([LEFT_LOBE] * 3)
    w = np.zeros((3, 3))
    b = np.array([50.0, -50.0, 3.0])
    steps = 50

    traj, u = rollout_numpy_batched(starts, w, b, steps=steps, u_max=10.0)

    assert (u[:, 0] == 10.0).all() and (u[:, 1] == -10.0).all()
    assert (u[:, 2] == 3.0).all()

    for lane, held in enumerate((10.0, -10.0, 3.0)):
        alone, _ = rollout_numpy(LEFT_LOBE, steps=steps, u=held)
        assert np.array_equal(traj[:, lane], alone), lane

    # torch clips and gates exactly as numpy does, the limits one per lane
    raw = np.array([50.0, -50.0, 3.0, 50.0])
    x = np.array([-8.0, -8.0, -8.0, 8.0])
    caps, gates = np.array([10.0, 10.0, np.inf, 10.0]), np.full(4, 2.0)
    mine = limit_control(torch.tensor(raw), torch.tensor(x), gates, caps).numpy()

    assert np.array_equal(mine, limit_control(raw, x, gates, caps))
    assert np.array_equal(mine, [10.0, -10.0, 3.0, 0.0])


def test_a_clipped_window_leaves_its_law_untrained():
    """A law held at the cap gets no gradient, so the window leaves it where it was."""
    w0 = torch.zeros(1, 3, dtype=torch.float64)
    b0 = torch.full((1,), 50.0, dtype=torch.float64)

    run = train_receding_horizon(
        state0=LEFT_LOBE, window=1.0, total_horizon=1.0, init=(w0, b0), **RECEDING
    ).lane(0)

    assert (run.u == 10.0).all()
    assert np.array_equal(run.w[0], w0.numpy()[0])
    assert np.array_equal(run.b[0], b0.numpy()[0])
    assert (run.grad_norm == 0).all()


def test_caps_batch_per_lane():
    """Each lane is capped by its own u_max, remainder branches included."""
    # u = 0.1·(-8 - 8 + 27) + 1 = 2.1 at the start, over the tighter cap only
    init = (torch.full((2, 3), 0.1, dtype=torch.float64), torch.ones(2, dtype=torch.float64))
    # this start is one where the recorded loss is off by an ulp batched,
    # capped or not
    batched = _two_limits(
        [2.0, 1.5],
        "u_max",
        [0.5, np.inf],
        state0=LEFT_LOBE,
        init=init,
        history_atol=1e-15,
    )

    # the capped lane sits at its cap, while the uncapped one trained freely
    assert batched[0][0].u[0] == 0.5 and batched[1][0].u[0] != 0.5
    assert (np.abs(batched[0][0].u) <= 0.5).all()


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]

    for test in tests:
        print(f"{test.__name__} ...")
        test()

    print(f"\n{len(tests)} passed" + ("" if CUDA else " (CUDA tests skipped)"))
