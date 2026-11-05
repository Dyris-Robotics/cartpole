# MuJoCo's compiled bindings (MjModel, mj_step, ...) ship without type stubs.
# pyright: reportAttributeAccessIssue=false

from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from cartpole_rl.interfaces.cartpole_command import CartPoleCommand
from cartpole_rl.interfaces.cartpole_interface import CartPoleInterface
from cartpole_rl.interfaces.cartpole_state import CartPoleState

MODEL_FILE = str(Path(__file__).parent.parent / "assets" / "cartpole.xml")


class MujocoCartPole(CartPoleInterface):
    """The cart-pole simulated in MuJoCo, with optional sensor noise and delay."""

    def __init__(
        self,
        model_file: str = MODEL_FILE,
        sensor_noise: dict[str, float] | None = None,
        action_delay_steps: int = 0,
    ) -> None:

        self.model = mujoco.MjModel.from_xml_path(model_file)
        self.data = mujoco.MjData(self.model)

        # State
        self.cart_pos_adr = self.model.sensor("cart_pos_sensor").adr[0]
        self.cart_vel_adr = self.model.sensor("cart_velocity_sensor").adr[0]
        self.pole_angle_adr = self.model.sensor("pole_angle_sensor").adr[0]
        self.pole_vel_adr = self.model.sensor("pole_velocity_sensor").adr[0]

        # Command
        self.cart_motor_id = self.model.actuator("cart_motor").id

        self.sensor_noise = sensor_noise or {}  # std per value, missing = 0
        self.action_delay_steps = action_delay_steps
        self.rng = np.random.default_rng()  # replaced by the env's rng in reset()
        self.cart_motor_command_queue: deque[CartPoleCommand] = deque()

    def read_data(self) -> CartPoleState:
        """What the sensors report: the true state plus noise."""
        state = self.read_true_state()
        return CartPoleState(
            cart_pos=state.cart_pos + self._noise("cart_pos"),
            cart_vel=state.cart_vel + self._noise("cart_vel"),
            pole_angle=state.pole_angle + self._noise("pole_angle"),
            pole_vel=state.pole_vel + self._noise("pole_vel"),
        )

    def apply_command(self, command: CartPoleCommand) -> None:
        """Queue the command and send the oldest one (action delay) to the motor."""
        self.cart_motor_command_queue.append(command)
        oldest = self.cart_motor_command_queue.popleft()
        self.data.ctrl[self.cart_motor_id] = oldest.cart_motor_command

    def read_true_state(self) -> CartPoleState:
        """The exact simulated state, without sensor noise."""
        return CartPoleState(
            cart_pos=float(self.data.sensordata[self.cart_pos_adr]),
            cart_vel=float(self.data.sensordata[self.cart_vel_adr]),
            pole_angle=float(self.data.sensordata[self.pole_angle_adr]),
            pole_vel=float(self.data.sensordata[self.pole_vel_adr]),
        )

    def advance(self, duration: float) -> None:
        """Let `duration` seconds of simulated time pass."""
        n_steps = round(duration / self.model.opt.timestep)
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

    def reset(self, rng: np.random.Generator, reset_noise: float = 0.0) -> None:
        """Hang the pole down, near rest, with small random offsets."""
        self.rng = rng
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = rng.uniform(-reset_noise, reset_noise, size=2)
        self.data.qvel[:] = rng.uniform(-reset_noise, reset_noise, size=2)
        mujoco.mj_forward(self.model, self.data)
        self.cart_motor_command_queue = deque(
            [CartPoleCommand(cart_motor_command=0.0)] * self.action_delay_steps
        )

    def kick_pole(self, velocity: float) -> None:
        """Instantly add angular velocity to the pole."""
        self.data.qvel[1] += velocity

    def set_cart_gear(self, cart_gear: float) -> None:
        """Change the strength of the cart motor."""
        self.model.actuator_gear[self.cart_motor_id, 0] = cart_gear

    def _noise(self, name: str) -> float:
        """A random sensor error for one value (0.0 if no noise is set for it)."""
        std = self.sensor_noise.get(name, 0.0)
        return float(self.rng.normal(0.0, std))
