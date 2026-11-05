import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cartpole_rl.disturbances.disturbance import Disturbance
from cartpole_rl.envs.observations import Observations
from cartpole_rl.envs.rewards import Rewards
from cartpole_rl.envs.terminations import Terminations
from cartpole_rl.interfaces.cartpole_command import CartPoleCommand
from cartpole_rl.interfaces.cartpole_interface import CartPoleInterface
from cartpole_rl.randomization.randomization import Randomization

CONTROL_PERIOD = 0.02
EPISODE_SECONDS = 10.0  # at most 10 s per episode = 500 steps
RESET_NOISE = 0.1


class Environment(gym.Env):
    """Gymnasium environment for the cart-pole swing-up, around any robot."""

    def __init__(
        self,
        system: CartPoleInterface,
        disturbances: list[Disturbance] | None = None,
        randomizations: list[Randomization] | None = None,
        configs: dict | None = None,
    ) -> None:

        # Configuration Init
        self.config_init(configs)

        # System
        self.system = system

        # Disturbances
        self.disturbances = disturbances or []

        # Randomizations
        self.randomizations = randomizations or []

        # Environment
        self.obs = Observations(**(self.config.get("observations") or {}))
        self.rewards = Rewards(**(self.config.get("rewards") or {}))
        self.terminations = Terminations(**(self.config.get("terminations") or {}))

        # Gymnasium requires these two.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=self.obs.low, high=self.obs.high, dtype=np.float32
        )

        # Timing
        self.max_steps = round(self.episode_seconds / self.control_period)
        self.steps = 0

        # Disturbances Intervals
        self.interval_disturbances = []
        for disturbance in self.disturbances:
            every_steps = round(disturbance.every_seconds / self.control_period)
            self.interval_disturbances.append((disturbance, every_steps))

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply one action and advance the robot by one control period."""
        # Applying Action
        action = np.clip(action, -1.0, 1.0)
        command = CartPoleCommand(cart_motor_command=float(action[0]))
        self.system.apply_command(command)

        # Applying Interval Disturbances
        for disturbance, every_steps in self.interval_disturbances:
            if self.steps > 0 and self.steps % every_steps == 0:
                disturbance.apply(self.system, self.np_random)

        # Step System
        self.system.advance(self.control_period)
        self.steps += 1

        # Observation, what the sensors report (can be noisy)
        obs = self.obs.get(self.system.read_data())

        # True state exactly as returned by the system
        true_state = self.system.read_true_state()

        # Reward
        reward, reward_parts = self.rewards.get(true_state, action)

        # Termination
        terminated = self.terminations.get(true_state)
        truncated = self.steps >= self.max_steps  # time's up

        return obs, reward, terminated, truncated, reward_parts

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Start a new episode and return the first observation."""
        super().reset(seed=seed)  # sets up self.np_random

        # Reset System
        self.system.reset(self.np_random, self.reset_noise)

        # Applying Randomization
        for randomization in self.randomizations:
            randomization.apply(self.system, self.np_random)

        self.obs.reset()
        self.rewards.reset()
        self.terminations.reset()
        self.steps = 0

        return self.obs.get(self.system.read_data()), {}

    def config_init(self, configs: dict | None = None) -> None:
        """Store the configs and read the `env` section."""
        self.config = configs or {}

        # Env
        env_cfg = self.config.get("env") or {}
        self.control_period = env_cfg.get("control_period", CONTROL_PERIOD)
        self.episode_seconds = env_cfg.get("episode_seconds", EPISODE_SECONDS)
        self.reset_noise = env_cfg.get("reset_noise", RESET_NOISE)
