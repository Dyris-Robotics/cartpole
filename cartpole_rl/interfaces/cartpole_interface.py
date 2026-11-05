from abc import ABC, abstractmethod

import numpy as np

from cartpole_rl.interfaces.cartpole_command import CartPoleCommand
from cartpole_rl.interfaces.cartpole_state import CartPoleState


class CartPoleInterface(ABC):
    """A cart-pole robot, simulated or real, as seen by the environment."""

    @abstractmethod
    def read_data(self) -> CartPoleState:
        """What the sensors report."""

    @abstractmethod
    def apply_command(self, command: CartPoleCommand) -> None:
        """Send a command to the motor."""

    @abstractmethod
    def advance(self, duration: float) -> None:
        """Let `duration` seconds pass."""

    @abstractmethod
    def reset(self, rng: np.random.Generator, reset_noise: float) -> None:
        """Bring the robot to a start position."""

    def read_true_state(self) -> CartPoleState:
        """The most accurate state available. By default, the sensor readings."""
        return self.read_data()

    # Optional capabilities, used by randomizations and disturbances. A robot
    # that cannot do them (e.g. real hardware) keeps these defaults.

    def kick_pole(self, velocity: float) -> None:
        """Instantly add angular velocity to the pole."""
        raise NotImplementedError(f"{type(self).__name__} cannot kick the pole")

    def set_cart_gear(self, cart_gear: float) -> None:
        """Change the strength of the cart motor."""
        raise NotImplementedError(f"{type(self).__name__} cannot change its gear")
