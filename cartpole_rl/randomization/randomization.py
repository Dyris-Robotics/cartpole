from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeVar

import numpy as np

from cartpole_rl.interfaces.cartpole_interface import CartPoleInterface

RandomizationT = TypeVar("RandomizationT", bound="Randomization")

RANDOMIZATIONS: dict[str, type["Randomization"]] = {}


def register_randomization(
    name: str,
) -> Callable[[type[RandomizationT]], type[RandomizationT]]:
    """Class decorator that makes a randomization available under `name` in configs."""

    def wrap(cls: type[RandomizationT]) -> type[RandomizationT]:
        RANDOMIZATIONS[name] = cls
        return cls

    return wrap


class Randomization(ABC):
    """A change applied to the robot at the start of every episode."""

    @abstractmethod
    def apply(self, system: CartPoleInterface, rng: np.random.Generator) -> None:
        """Randomize the robot once."""
