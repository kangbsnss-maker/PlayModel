"""Recorded conditional action masks reduce steering chatter without altering sent actions."""
import math
from playmodel.learning import MOVEMENTS, ALL_ACTIONS_MASK


class MovementStabilizer:
    def __init__(self, hold_ms: float = 200):
        if not 0 <= hold_ms <= 300:
            raise ValueError('Movement hold must be 0..300 ms')
        self.hold_ns = int(hold_ms*1e6)
        self.previous = 0
        self.changed_at = 0

    def mask(self, state, previous_action: int, now_ns: int):
        if previous_action != self.previous:
            self.previous, self.changed_at = previous_action, now_ns
        dx, dy, valid = state.navigation
        if not valid or self.hold_ns == 0:
            return ALL_ACTIONS_MASK
        length = math.hypot(dx,dy)
        if length < 1e-6:
            return ALL_ACTIONS_MASK
        scores = [0. if action == (0,0) else (action[0]*dx+action[1]*dy)/math.hypot(*action)/length
                  for action in MOVEMENTS]
        # A nearby threat behind us must not cancel every commitment interval.
        # Interrupt immediately when the held direction approaches a close threat.
        held_x, held_y = MOVEMENTS[previous_action]
        danger = bool(state.player and any(
            math.hypot((h.x-state.player[0])*16/9,h.y-state.player[1]) < .10+h.radius
            and ((h.x-state.player[0])*16/9*held_x+(h.y-state.player[1])*held_y >= 0)
            for h in state.hazards))
        if previous_action and now_ns-self.changed_at < self.hold_ns and scores[previous_action] >= .3 and not danger:
            return tuple(index == previous_action for index in range(len(MOVEMENTS)))
        best = max(scores[1:])
        return tuple(index > 0 and score >= best-.20 for index,score in enumerate(scores))
