import numpy as np

from cartpole_rl.disturbances.disturbance import Disturbance, register_disturbance
from cartpole_rl.interfaces.cartpole_interface import CartPoleInterface


@register_disturbance("pole_kick")
class PoleKick(Disturbance):
    """Knocks the pole at regular intervals with a random angular velocity."""

    def __init__(
        self, kick_angular_velocity: float = 1.0, every_seconds: float = 3.0
    ) -> None:
        super().__init__(every_seconds)
        self.kick_angular_velocity = kick_angular_velocity  # rad/s

    def apply(self, system: CartPoleInterface, rng: np.random.Generator) -> None:
        """Kick the pole by up to ±kick_angular_velocity rad/s."""
        system.kick_pole(
            rng.uniform(-self.kick_angular_velocity, self.kick_angular_velocity)
        )
