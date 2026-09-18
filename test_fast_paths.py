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
        iters=150,
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
        w_one, b_one, history_one, grad_norm_one = train_policy_batched(
            batch=1,
            effort_weight=[lam],
            learning_rate=[lr],
            use_graph=False,
            init=(w0[i : i + 1], b0[i : i + 1]),
            **common,
        )

        assert (w[i] - w_one[0]).abs().max() < 1e-12, (i, w[i], w_one[0])
        assert (b[i] - b_one[0]).abs().max() < 1e-12, (i, b[i], b_one[0])
        assert np.abs(history[:, i] - history_one[:, 0]).max() < 1e-15
        # the graphed lane must record the same norms the eager run did
        assert np.allclose(grad_norm[:, i], grad_norm_one[:, 0], rtol=1e-12, atol=0)

    # two lanes given identical settings do the identical arithmetic, so these
    # have no reassociation to excuse a difference and must match exactly
    assert torch.equal(w[1], w[3])
    assert torch.equal(b[1], b[3])


def test_mixed_windows_match_their_own_runs():
    """A short lane riding along a longer batch must equal a run at its length.

    Every lane here integrates to the longest window in the batch and is masked
    back to its own, so this is the check that the mask covers exactly the
    steps a standalone run would have taken -- one step out in either direction
    and the shorter lanes would drift.

    effort_horizon is left None on purpose. With the long moment rollout in
    play a last-bit difference is amplified into a visible one, which would
    make this a measurement of the attractor rather than of the mask.
    """
    if not CUDA:
        print("  (skipped: no CUDA device)")
        return

    horizons = [1.0, 1.25, 1.5, 1.75, 2.0]
    lams = [0.06, 0.08, 0.10, 0.12, 0.14]
    common = dict(
        iters=120,
        penalize_effort=True,
        effort_horizon=None,
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
            use_graph=False,
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

    effort_horizon is left None for the reason the mixed-window test gives.
    """
    starts = [(0.0, 1.0, 1.05), (0.5, 1.0, 1.05), (1.0, 1.0, 1.05), (-0.3, 0.2, 2.0)]
    common = dict(
        horizon=1.0,
        iters=120,
        penalize_effort=True,
        effort_horizon=None,
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


def test_a_batch_may_not_mix_the_two_penalties():
    """Pathwise and frozen-moment penalties are different objectives."""
    try:
        train_policy_batched(
            batch=2,
            horizon=[1.0, 2.0],
            effort_horizon=1.0,
            penalize_effort=True,
            iters=2,
            device="cpu",
        )
    except ValueError:
        return

    raise AssertionError("a mixed batch should have been refused")


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


RECEDING = dict(
    iters=40,
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
        state0=start, horizon=1.0, effort_horizon=None, init=(w0, b0), **RECEDING
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
        effort_horizon=None,
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


def test_a_remainder_window_is_shorter():
    """A horizon the window doesn't divide ends on a short window, not an overshoot."""
    run = train_receding_horizon(window=1.0, total_horizon=2.5, **{**RECEDING, "iters": 2})

    assert run.windows.tolist() == [3]
    assert len(run.traj) == round(2.5 / (LYAPUNOV_EXP * DT)) + 1
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


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]

    for test in tests:
        print(f"{test.__name__} ...")
        test()

    print(f"\n{len(tests)} passed" + ("" if CUDA else " (CUDA tests skipped)"))
