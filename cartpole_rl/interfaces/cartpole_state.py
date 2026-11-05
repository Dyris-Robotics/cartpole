from dataclasses import dataclass


@dataclass
class CartPoleState:
    """The state of the cart-pole."""

    cart_pos: float  # m, 0 = middle of the rail
    cart_vel: float  # m/s
    pole_angle: float  # rad, 0 = hanging down, pi = straight up
    pole_vel: float  # rad/s
