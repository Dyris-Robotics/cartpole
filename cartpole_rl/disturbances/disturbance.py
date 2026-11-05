from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeVar

import numpy as np

from cartpole_rl.interfaces.cartpole_interface import CartPoleInterface

DisturbanceT = TypeVar("DisturbanceT", bound="Disturbance")

DISTURBANCES: dict[str, type["Disturbance"]] = {}


def register_disturbance(
    name: str,
) -> Callable[[type[DisturbanceT]], type[DisturbanceT]]:
    """Class decorator that makes a disturbance available under `name` in configs."""

    def wrap(cls: type[DisturbanceT]) -> type[DisturbanceT]:
        DISTURBANCES[name] = cls
        return cls

    return wrap


class Disturbance(ABC):
    """An event applied to the robot every `every_seconds` during an episode."""

    def __init__(self, every_seconds: float) -> None:
        self.every_seconds = every_seconds

    @abstractmethod
    def apply(self, system: CartPoleInterface, rng: np.random.Generator) -> None:
        """Disturb the robot once."""
