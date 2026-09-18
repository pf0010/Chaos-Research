"""Receding-horizon control of the Lorenz system, and plots of the result.

    python run_control.py -th 1 -ph 10 -i 100  # 10 windows of 1 LT, random start
    python run_control.py -ic 0 1 1.05         # from a given start instead
    python run_control.py -seed 3              # or from a reproducible random one
    python run_control.py -loss -s             # also save a loss curve
    python run_control.py -gn -s               # and a gradient norm vs. iteration
    python run_control.py -sg -s               # ∂L/∂state through the window
    python run_control.py -lgh 8               # loss-gradient vs. horizon

The projected horizon -ph is cut into windows of -th Lyapunov times. Each
window trains its own feedback law for -i iterations from where the previous
window's law left the system, starting from that law; the trajectory stitched
from those windows is what is scored and drawn.

The start is -ic if given, otherwise a random point on the attractor drawn from
-seed (itself drawn and printed when not given, so any run can be repeated).

Plotting a finished sweep is sweep.py's job: `python sweep.py -sc`.

This is the single-run door: one run, trained as a batch of one. A grid no
longer comes through here -- sweep.py calls the batched trainer directly, since
running the points as separate processes is exactly what made a sweep slow.
The two share params.py, so a flag added below should be added there too.
"""

import argparse

from figures import (
    plot_grad_norm_curve,
    plot_loss_curve,
    plot_loss_gradient_vs_horizon,
    plot_run_summary,
    plot_state_gradient_through_window,
)
from lorenz import euler_step, rk4_step
from params import DEFAULT_EFFORT_WEIGHT, SWEEPABLE, resolve, resolve_start
from training import train_policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-ic",
        "--initial_condition",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="start here (default: a random point on the attractor)",
    )
    parser.add_argument(
        "-seed",
        "--seed",
        type=int,
        default=None,
        help="seed for the random start (default: drawn, and printed)",
    )
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.05)
    parser.add_argument("-th", "--train_horizon", type=float, default=1)
    parser.add_argument("-ph", "--plot_horizon", type=float, default=100)
    parser.add_argument("-i", "--iters", type=int, default=600)
    parser.add_argument(
        "-lam", "--effort_weight", type=float, default=DEFAULT_EFFORT_WEIGHT
    )
    parser.add_argument("-lgh", "--loss_gradient_horizon", type=float, default=0)
    parser.add_argument(
        "-o",
        "--out_dir",
        default=None,
        help="where -s writes (default: ./plots/run_iters<iters>); sweep.py "
        "points every grid point at its own sweep directory",
    )
    parser.add_argument(
        "-nk",
        "--name_keys",
        default=None,
        metavar="NAMES",
        help="comma-separated parameters to put in the saved filename "
        "(default: all of them); sweep.py passes the axes it varied, so the "
        "constants stay in the sweep manifest instead of in every name",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="where the training runs (default: cuda when there is one)",
    )

    flags = parser.add_argument_group(title="Flags")
    flags.add_argument("-s", "--save", action="store_true")
    flags.add_argument(
        "--fp64",
        action="store_true",
        help="force float64; the GPU default is float32 (see kernels.py)",
    )
    flags.add_argument(
        "--no_graph", "--no-graph", dest="no_graph", action="store_true"
    )
    flags.add_argument(
        "-npe",
        "--no_penalize_effort",
        action="store_true",
        help="train on the task term alone; λ then has no effect on training",
    )
    flags.add_argument(
        "-euler",
        "--euler",
        action="store_true",
        help="integrate with forward euler instead of the default rk4",
    )
    flags.add_argument("-loss", "--loss_curve", action="store_true")
    flags.add_argument(
        "-gn",
        "--grad_norm_curve",
        action="store_true",
        help="also plot the gradient norm against iteration",
    )
    flags.add_argument(
        "-sg",
        "--state_gradient",
        action="store_true",
        help="also plot the gradient w.r.t. each state in the window, across training",
    )
    args = parser.parse_args()

    integrator = euler_step if args.euler else rk4_step
    penalize_effort = not args.no_penalize_effort
    name_keys = (
        [resolve(name).name for name in args.name_keys.split(",") if name.strip()]
        if args.name_keys is not None
        else SWEEPABLE
    )

    if args.loss_gradient_horizon:
        plot_loss_gradient_vs_horizon(
            state0=args.initial_condition or [0, 1, 1.05],
            max_horizon=args.loss_gradient_horizon,
            integrator=integrator,
        )
    else:
        import torch

        state0, seed = resolve_start(
            {"initial_condition": args.initial_condition, "seed": args.seed}
        )
        x, y, z = state0
        print(
            f"start ({x:+.3f}, {y:+.3f}, {z:+.3f})  "
            + ("[given]" if seed is None else f"[random, seed {seed}]")
        )

        run = train_policy(
            state0=state0,
            lr=args.learning_rate,
            horizon=args.train_horizon,
            total_horizon=args.plot_horizon,
            iters=args.iters,
            penalize_effort=penalize_effort,
            effort_weight=args.effort_weight,
            integrator=integrator,
            record_params=args.state_gradient,
            device=args.device,
            dtype=torch.float64 if args.fp64 else None,
            use_graph=not args.no_graph,
        )

        shared = dict(
            state0=state0,
            seed=seed,
            lr=args.learning_rate,
            train_horizon=args.train_horizon,
            plot_horizon=args.plot_horizon,
            iters=args.iters,
            penalize_effort=penalize_effort,
            effort_weight=args.effort_weight,
            integrator=integrator,
            save=args.save,
            out_dir=args.out_dir,
            name_keys=name_keys,
        )

        if args.loss_curve:
            plot_loss_curve(run.history, **shared)

        if args.grad_norm_curve:
            plot_grad_norm_curve(run.grad_norm, **shared)

        if args.state_gradient:
            plot_state_gradient_through_window(
                run.params_seen, starts=run.starts, **shared
            )

        # scored and drawn on the stitched trajectory: law k over window k
        plot_run_summary(
            (torch.as_tensor(run.w[-1]), torch.as_tensor(run.b[-1])),
            traj=run.traj,
            u=run.u,
            **shared,
        )
