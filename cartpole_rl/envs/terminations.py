from cartpole_rl.interfaces.cartpole_state import CartPoleState

CART_POS_LIMIT_DEFAULT = 1.3


class Terminations:
    """Game over rules: the cart past the limit."""

    def __init__(self, cart_pos_limit: float = CART_POS_LIMIT_DEFAULT) -> None:
        self.cart_pos_limit = cart_pos_limit

    def get(self, state: CartPoleState) -> bool:
        """True when the cart is beyond the position limit."""
        return abs(state.cart_pos) > self.cart_pos_limit

    def reset(self) -> None:
        """Clear per-episode memory (none for now)."""
