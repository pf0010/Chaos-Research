"""Train a feedback law on the Lorenz system and plot the result.

    python run_control.py -th 1.0 -i 600 -pe   # train, then show the summary
    python run_control.py -loss -s             # also save a loss curve
    python run_control.py -lgh 8               # loss-gradient vs. horizon

Plotting a finished sweep is sweep.py's job: `python sweep.py -sc`.

sweep.py drives this module over a grid, so the success line plot_run_summary
prints and the short flags below are both part of its interface.
"""

import argparse

from figures import plot_loss_curve, plot_loss_gradient_vs_horizon, plot_run_summary
from lorenz import euler_step, rk4_step
from training import DEFAULT_EFFORT_WEIGHT, train_policy

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

    flags = parser.add_argument_group(title="Flags")
    flags.add_argument("-s", "--save", action="store_true")
    flags.add_argument("-pe", "--penalize_effort", action="store_true")
    flags.add_argument("-rk4", "--rk4", action="store_true")
    flags.add_argument("-loss", "--loss_curve", action="store_true")
    args = parser.parse_args()

    integrator = rk4_step if args.rk4 else euler_step

    if args.loss_gradient_horizon:
        plot_loss_gradient_vs_horizon(
            state0=args.initial_condition,
            max_horizon=args.loss_gradient_horizon,
            integrator=integrator,
        )
    else:
        history = []
        params = train_policy(
            state0=args.initial_condition,
            lr=args.learning_rate,
            horizon=args.train_horizon,
            iters=args.iters,
            penalize_effort=args.penalize_effort,
            effort_weight=args.effort_weight,
            integrator=integrator,
            history=history,
        )

        if args.loss_curve:
            plot_loss_curve(
                history,
                state0=args.initial_condition,
                lr=args.learning_rate,
                train_horizon=args.train_horizon,
                iters=args.iters,
                penalize_effort=args.penalize_effort,
                effort_weight=args.effort_weight,
                integrator=integrator,
                save=args.save,
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
        )
