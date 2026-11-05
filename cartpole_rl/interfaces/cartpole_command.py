from dataclasses import dataclass


@dataclass
class CartPoleCommand:
    """The Commands of the cart-pole."""

    cart_motor_command: float  # -1..1
