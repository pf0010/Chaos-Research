"""Train a feedback law on the Lorenz system and plot the result.

    python run_control.py -th 1.0 -i 600 -pe   # train, then show the summary
    python run_control.py -loss -s             # also save a loss curve
    python run_control.py -lgh 8               # loss-gradient vs. horizon

Plotting a finished sweep is sweep.py's job: `python sweep.py -sc`.

This is the single-run door: one policy, trained as a batch of one. A grid no
longer comes through here -- sweep.py calls the batched trainer directly, since
running the points as separate processes is exactly what made a sweep slow.
The two share params.py, so a flag added below should be added there too.
"""

import argparse

from figures import plot_loss_curve, plot_loss_gradient_vs_horizon, plot_run_summary
from lorenz import euler_step, rk4_step
from params import DEFAULT_EFFORT_WEIGHT, SWEEPABLE, resolve
from training import train_policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-ic",
        "--initial_condition",
        nargs=3,
        type=float,
        default=[0, 1, 1.05],
        metavar=("X", "Y", "Z"),
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
    flags.add_argument("-pe", "--penalize_effort", action="store_true")
    flags.add_argument(
        "-euler",
        "--euler",
        action="store_true",
        help="integrate with forward euler instead of the default rk4",
    )
    flags.add_argument("-loss", "--loss_curve", action="store_true")
    args = parser.parse_args()

    integrator = euler_step if args.euler else rk4_step
    name_keys = (
        [resolve(name).name for name in args.name_keys.split(",") if name.strip()]
        if args.name_keys is not None
        else SWEEPABLE
    )

    if args.loss_gradient_horizon:
        plot_loss_gradient_vs_horizon(
            state0=args.initial_condition,
            max_horizon=args.loss_gradient_horizon,
            integrator=integrator,
        )
    else:
        import torch

        history = []
        params = train_policy(
            state0=args.initial_condition,
            lr=args.learning_rate,
            horizon=args.train_horizon,
            effort_horizon=args.plot_horizon,
            iters=args.iters,
            penalize_effort=args.penalize_effort,
            effort_weight=args.effort_weight,
            integrator=integrator,
            history=history,
            device=args.device,
            dtype=torch.float64 if args.fp64 else None,
            use_graph=not args.no_graph,
        )

        if args.loss_curve:
            plot_loss_curve(
                history,
                state0=args.initial_condition,
                lr=args.learning_rate,
                train_horizon=args.train_horizon,
                plot_horizon=args.plot_horizon,
                iters=args.iters,
                penalize_effort=args.penalize_effort,
                effort_weight=args.effort_weight,
                integrator=integrator,
                save=args.save,
                out_dir=args.out_dir,
                name_keys=name_keys,
            )

        plot_run_summary(
            params,
            state0=args.initial_condition,
            lr=args.learning_rate,
            train_horizon=args.train_horizon,
            plot_horizon=args.plot_horizon,
            iters=args.iters,
            penalize_effort=args.penalize_effort,
            save=args.save,
            effort_weight=args.effort_weight,
            integrator=integrator,
            out_dir=args.out_dir,
            name_keys=name_keys,
        )
