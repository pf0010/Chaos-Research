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
from lorenz import euler_step, lorenz_rhs, rk4_step
from training import BatchedAdam, effort_moments, train_policy_batched

CUDA = cuda_available()


def test_rhs_is_batch_polymorphic():
    """A batch of states must step exactly as each state would alone."""
    rng = np.random.default_rng(0)
    states = rng.normal(0, 10, (8, 3))
    us = rng.normal(0, 5, 8)

    batched = lorenz_rhs(states, us)

    for i, (state, u) in enumerate(zip(states, us)):
        assert np.array_equal(lorenz_rhs(state, u), batched[i])


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

    w, b, history = train_policy_batched(
        batch=4, effort_weight=lams, learning_rate=lrs, use_graph=True, **common
    )

    for i, (lr, lam) in enumerate(zip(lrs, lams)):
        w_one, b_one, history_one = train_policy_batched(
            batch=1,
            effort_weight=[lam],
            learning_rate=[lr],
            use_graph=False,
            **common,
        )

        assert (w[i] - w_one[0]).abs().max() < 1e-12, (i, w[i], w_one[0])
        assert (b[i] - b_one[0]).abs().max() < 1e-12, (i, b[i], b_one[0])
        assert np.abs(history[:, i] - history_one[:, 0]).max() < 1e-15

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

    w, b, _ = train_policy_batched(
        batch=5, horizon=horizons, effort_weight=lams, use_graph=True, **common
    )

    for i, (horizon, lam) in enumerate(zip(horizons, lams)):
        w_one, b_one, _ = train_policy_batched(
            batch=1,
            horizon=horizon,
            effort_weight=[lam],
            use_graph=False,
            **common,
        )

        assert (w[i] - w_one[0]).abs().max() < 1e-12, (horizon, w[i], w_one[0])
        assert (b[i] - b_one[0]).abs().max() < 1e-12, (horizon, b[i], b_one[0])


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


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]

    for test in tests:
        print(f"{test.__name__} ...")
        test()

    print(f"\n{len(tests)} passed" + ("" if CUDA else " (CUDA tests skipped)"))
