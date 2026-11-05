"""Builds a ready-to-use environment around any robot implementing CartPoleInterface."""

from cartpole_rl.disturbances.disturbance import DISTURBANCES, Disturbance
from cartpole_rl.envs.environment import Environment
from cartpole_rl.interfaces.cartpole_interface import CartPoleInterface
from cartpole_rl.randomization.randomization import RANDOMIZATIONS, Randomization

KNOWN_SECTIONS = {
    "env",
    "system",
    "observations",
    "rewards",
    "terminations",
    "randomizations",
    "disturbances",
}


def _check_names(names: set, known: set, what: str) -> None:
    """Crash on any name the code does not know (catches typos in the yaml)."""
    unknown = names - known
    if unknown:
        raise ValueError(f"Unknown {what}: {unknown}")


def make_randomizations(configs: dict) -> list[Randomization]:
    """Randomizations applied at each reset (missing or empty section = none)."""
    section = configs.get("randomizations") or {}
    _check_names(set(section), set(RANDOMIZATIONS), "randomizations")
    return [RANDOMIZATIONS[name](**params) for name, params in section.items()]


def make_disturbances(configs: dict) -> list[Disturbance]:
    """Disturbances applied at intervals in an episode (missing section = none)."""
    section = configs.get("disturbances") or {}
    _check_names(set(section), set(DISTURBANCES), "disturbances")
    return [DISTURBANCES[name](**params) for name, params in section.items()]


def make_env(system: CartPoleInterface, configs: dict | None = None) -> Environment:
    """Build the environment around the robot you give it."""
    configs = configs or {}
    _check_names(set(configs), KNOWN_SECTIONS, "config sections")

    return Environment(
        system,
        disturbances=make_disturbances(configs),
        randomizations=make_randomizations(configs),
        configs=configs,
    )
