"""Common controller base class.

Every controller is a callable  u = controller(state, t).
The cart reference x_ref(t) is set by the experiment (scenario S3 changes
it); controllers only ever see the current reference, never the future one.
"""


class Controller:
    name = "base"

    def __init__(self):
        self.x_ref = lambda t: 0.0

    def reset(self):
        """Clear internal state (integrators, warm starts) before an episode."""

    def __call__(self, state, t):
        raise NotImplementedError
