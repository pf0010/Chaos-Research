"""The one place a run parameter is defined.

Every parameter a sweep can vary is described once here: the short key that
names files, the run_control.py flag that sets it, how it is written into a
filename, and how it is shown to a reader. Filenames, the sweep csv, the
manifest and the figure captions all derive from this table, so adding a
parameter is a single edit rather than four consistent ones.

Filenames built from it look like

    lam0p100_th1p000_reg-on_int-euler.png

Fixed decimal places so a directory sorts in numeric order, `p` instead of `.`
so the name survives \\includegraphics, and an explicit token for every value so
`int-euler` is distinguishable from a run that predates the flag.

Nothing here imports torch or matplotlib; sweep.py leans on that.
"""

from dataclasses import dataclass
from typing import Any, Callable

DEFAULT_EFFORT_WEIGHT = 0.07

TRUE_WORDS = {"1", "true", "on", "yes"}
FALSE_WORDS = {"0", "false", "off", "no"}


def decimal_tag(value, places=3):
    """0.1 -> '0p100'. Fixed width sorts; no dot keeps LaTeX happy."""
    return f"{float(value):.{places}f}".replace(".", "p")


def _decimal(places):
    return lambda value: decimal_tag(value, places)


def parse_bool(text):
    lowered = str(text).strip().lower()

    if lowered in TRUE_WORDS:
        return True
    if lowered in FALSE_WORDS:
        return False

    # a numeric axis (`--sweep rk4=0:1:1`) arrives as a float
    try:
        return bool(float(lowered))
    except ValueError:
        raise ValueError(f"expected a boolean, got {text!r}") from None


def _parse_vector(text):
    if isinstance(text, (list, tuple)):
        return tuple(text)

    return tuple(float(part) for part in str(text).split("|"))


@dataclass(frozen=True)
class Param:
    name: str  # canonical: the csv column and the run_control.py dest
    key: str  # short: what appears in a filename
    flag: str  # run_control.py's short flag
    label: str  # for captions and axis titles
    default: Any
    parse: Callable[[Any], Any]  # text -> value, for the cli and the csv
    fmt: Callable[[Any], str]  # value -> filename token body
    show: Callable[[Any], str] = str  # value -> caption text
    sweepable: bool = True
    store_true: bool = False  # set by the flag's presence, not by a value
    nargs: int = 1
    # whether a sweep can vary this *within* one batched training run. Only a
    # parameter that leaves the shape of the computation alone can: it rides
    # along as a per-element tensor. Anything that changes the step count or
    # the integrator splits the grid into separate runs instead.
    batchable: bool = False


PARAMS = (
    Param(
        "initial_condition",
        "ic",
        "-ic",
        "initial condition",
        (0, 1, 1.05),
        _parse_vector,
        lambda v: "-".join(decimal_tag(c) for c in v),
        show=lambda v: f"({','.join(str(c) for c in v)})",
        sweepable=False,
        nargs=3,
    ),
    Param(
        "learning_rate",
        "lr",
        "-lr",
        "learning rate",
        0.05,
        float,
        _decimal(4),
        batchable=True,
    ),
    Param(
        "train_horizon",
        "th",
        "-th",
        "training window (Lyapunov times)",
        1.0,
        float,
        _decimal(3),
    ),
    Param(
        "plot_horizon",
        "ph",
        "-ph",
        "plotting window (Lyapunov times)",
        100.0,
        float,
        _decimal(1),
    ),
    Param(
        "iters",
        "iters",
        "-i",
        "training iterations",
        600,
        lambda v: int(float(v)),
        lambda v: f"{int(v):d}",
    ),
    Param(
        "effort_weight",
        "lam",
        "-lam",
        "λ (control-effort penalty)",
        DEFAULT_EFFORT_WEIGHT,
        float,
        _decimal(3),
        batchable=True,
    ),
    Param(
        "penalize_effort",
        "reg",
        "-pe",
        "effort penalty",
        False,
        parse_bool,
        lambda v: "on" if v else "off",
        show=lambda v: "on" if v else "off",
        store_true=True,
    ),
    Param(
        "rk4",
        "int",
        "-rk4",
        "integrator",
        False,
        parse_bool,
        lambda v: "rk4" if v else "euler",
        show=lambda v: "on" if v else "off",
        store_true=True,
    ),
)

BY_NAME = {param.name: param for param in PARAMS}
BY_KEY = {param.key: param for param in PARAMS}
SWEEPABLE = [param.name for param in PARAMS if param.sweepable]
BATCHABLE = [param.name for param in PARAMS if param.batchable]
# what a batched run has to hold fixed, and so what sweep.py groups the grid by
SHAPING = [param.name for param in PARAMS if not param.batchable]


def resolve(name):
    """Accept either the canonical name or the short key."""
    name = name.strip()

    if name in BY_NAME:
        return BY_NAME[name]
    if name in BY_KEY:
        return BY_KEY[name]

    known = ", ".join(f"{p.name} ({p.key})" for p in PARAMS)
    raise SystemExit(f"unknown parameter {name!r}; known parameters: {known}")


def defaults():
    return {param.name: param.default for param in PARAMS}


def run_values(**overrides):
    """A full parameter dict: the defaults with whatever the caller knows."""
    values = defaults()
    values.update({k: v for k, v in overrides.items() if v is not None})

    return values


def token(name, value):
    param = BY_NAME[name]
    text = param.fmt(value)
    # a word value gets a dash so `reg-on` reads; a number doesn't need one,
    # since the first digit already marks where the key ends
    separator = "" if text[:1].isdigit() else "-"

    return f"{param.key}{separator}{text}"


def stem(values, keys):
    """'lam0p100_th1p000' — the filename body for one run.

    `keys` is what varies across the batch this run belongs to; anything held
    constant stays out of the name and goes in the sweep manifest instead.
    Sorted by short key, so the same parameter set always yields one name.
    """
    keys = [k if k in BY_NAME else resolve(k).name for k in keys]
    missing = [k for k in keys if k not in values]

    if missing:
        raise KeyError(f"no value for {', '.join(missing)}")

    ordered = sorted(keys, key=lambda name: BY_NAME[name].key)

    return "_".join(token(name, values[name]) for name in ordered) or "run"


def shown(values, names=None):
    """{name: caption text} in registry order, for settings_caption."""
    names = names if names is not None else [p.name for p in PARAMS]

    return {name: BY_NAME[name].show(values[name]) for name in names if name in values}


def csv_value(name, value):
    param = BY_NAME[name]

    if param.nargs > 1:
        return "|".join(str(c) for c in value)
    if param.store_true:
        return int(bool(value))

    return value


def frange(start, stop, step):
    """Inclusive of stop, with a tolerance so 0.15 isn't dropped by float error."""
    if step <= 0:
        raise SystemExit(f"step must be positive, got {step}")

    values = []
    n = 0

    while True:
        value = round(start + n * step, 6)

        if value > stop + step / 2:
            break

        values.append(value)
        n += 1

    return values


def parse_axis(spec):
    """'lam=0.05:0.15:0.01' or 'rk4=0,1' -> (canonical name, [values])."""
    name, sep, rhs = spec.partition("=")

    if not sep or not rhs.strip():
        raise SystemExit(f"expected NAME=SPEC, got {spec!r}")

    param = resolve(name)

    if not param.sweepable:
        raise SystemExit(f"{param.name} cannot be swept")

    if ":" in rhs:
        parts = rhs.split(":")

        if len(parts) != 3:
            raise SystemExit(f"expected NAME=start:stop:step, got {spec!r}")

        try:
            start, stop, step = (float(part) for part in parts)
        except ValueError:
            raise SystemExit(f"non-numeric range in {spec!r}") from None

        raw = frange(start, stop, step)
    else:
        raw = [part for part in rhs.split(",") if part.strip()]

    try:
        values = [param.parse(value) for value in raw]
    except ValueError as err:
        raise SystemExit(f"bad value in {spec!r}: {err}") from None

    if not values:
        raise SystemExit(f"{spec!r} produced no values")

    # a range that steps past its own resolution would name two runs the same
    tags = [param.fmt(value) for value in values]

    if len(set(tags)) != len(tags):
        raise SystemExit(
            f"{spec!r} has values that format to the same filename token "
            f"(e.g. {sorted(t for t in tags if tags.count(t) > 1)[0]}); "
            f"widen the step or give {param.name} more decimal places in params.py"
        )

    return param.name, values
