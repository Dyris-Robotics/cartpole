import math

import numpy as np

from cartpole_rl.interfaces.cartpole_state import CartPoleState

EFFORT_COST_DEFAULT = 0.01
ACTION_RATE_COST_DEFAULT = 0.0


class Rewards:
    """How good one step was: upright - effort."""

    def __init__(
        self,
        effort_cost: float = EFFORT_COST_DEFAULT,
        action_rate_cost: float = ACTION_RATE_COST_DEFAULT,
    ) -> None:
        self.effort_cost = effort_cost
        self.action_rate_cost = action_rate_cost
        self.prev_action = 0.0

    def get(self, state: CartPoleState, action: np.ndarray) -> tuple[float, dict]:
        """Return the reward of one step and its parts, for logging."""
        upright = (1 - math.cos(state.pole_angle)) / 2  # 0 hanging down, 1 straight up
        effort = self.effort_cost * float(action[0]) ** 2

        # Big change = jerky
        rate = self.action_rate_cost * float(action[0] - self.prev_action) ** 2
        self.prev_action = float(action[0])  # remember it for the next step

        reward = upright - effort - rate

        return reward, {"upright": upright, "effort": effort, "action_rate": rate}

    def reset(self) -> None:
        """Forget the previous action at the start of an episode."""
        self.prev_action = 0.0
