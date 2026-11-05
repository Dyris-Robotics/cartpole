import numpy as np

from cartpole_rl.interfaces.cartpole_state import CartPoleState

CART_POS_MAX_DEFAULT = 1.5
CART_VEL_MAX_DEFAULT = 10.0
POLE_VEL_MAX_DEFAULT = 30.0


class Observations:
    """What the agent sees: [cart_pos, cos(angle), sin(angle), cart_vel, pole_vel]."""

    def __init__(
        self,
        cart_pos_max: float = CART_POS_MAX_DEFAULT,
        cart_vel_max: float = CART_VEL_MAX_DEFAULT,
        pole_vel_max: float = POLE_VEL_MAX_DEFAULT,
    ) -> None:
        # Smallest and largest value of each number, in the order get() returns them.
        #                     cart_pos      cos  sin  cart_vel      pole_vel
        self.high = np.array(
            [cart_pos_max, 1.0, 1.0, cart_vel_max, pole_vel_max], dtype=np.float32
        )
        self.low = -self.high

    def get(self, state: CartPoleState) -> np.ndarray:
        """Build the observation from a robot state, clipped to its bounds."""
        obs = np.array(
            [
                state.cart_pos,
                np.cos(state.pole_angle),
                np.sin(state.pole_angle),
                state.cart_vel,
                state.pole_vel,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, self.low, self.high)

    def reset(self) -> None:
        """Clear per-episode memory (none for now)."""
