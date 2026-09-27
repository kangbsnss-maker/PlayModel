"""Bounded non-confirming focus acquisition on independently observed cards."""
from dataclasses import dataclass, field


@dataclass
class LevelUpFocusRecovery:
    previous: tuple | None = None
    attempted: set = field(default_factory=set)

    def observe(self, selection, *, scene, decision_id, frame_id, observed_at_ns, now_ns):
        ratios = dict(selection.ratios)
        if (scene != 'level_up' or not decision_id or decision_id in self.attempted
                or selection.reason != 'no_selected_candidate'
                or not 0 < observed_at_ns <= now_ns <= observed_at_ns + 750_000_000
                or not all(.05 <= ratios.get(f'choose_{i}', 0) < .5 for i in range(4))):
            self.previous = None
            return None
        current = (decision_id, frame_id, observed_at_ns)
        previous, self.previous = self.previous, current
        if (previous is None or previous[0] != decision_id or previous[1] == frame_id
                or previous[2] >= observed_at_ns):
            return 'wait'
        # Left cannot buy, reroll or accept a card. Never infer which card gets
        # focus: the normal pixel check must succeed on the next observation.
        return 'left'

    def mark_sent(self, decision_id):
        self.attempted.add(decision_id)
        self.previous = None
