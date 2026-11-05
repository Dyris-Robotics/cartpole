import numpy as np

from cartpole_rl.interfaces.cartpole_interface import CartPoleInterface
from cartpole_rl.randomization.randomization import (
    Randomization,
    register_randomization,
)


@register_randomization("cart_gear")
class CartGearRandom(Randomization):
    """Gives the cart motor a random strength at every reset."""

    def __init__(self, gear_min: float = 13.0, gear_max: float = 17.0) -> None:
        self.gear_min = gear_min
        self.gear_max = gear_max

    def apply(self, system: CartPoleInterface, rng: np.random.Generator) -> None:
        """Set a gear drawn uniformly between gear_min and gear_max."""
        system.set_cart_gear(rng.uniform(self.gear_min, self.gear_max))
