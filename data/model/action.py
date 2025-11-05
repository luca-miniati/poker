from dataclasses import dataclass
from typing import Optional


@dataclass
class Action:
    hand_id: str
    actor: str
    action_index: int
    action_type: str
    amount: float = 0
    total_pot_amount: float = 0
    street_index: int = 0
    cards: Optional[str] = None
    is_terminal: bool = False
    raw_action: Optional[str] = None

    @property
    def player_idx(self) -> Optional[int]:
        '''
        Returns the 0-indexed player index if the actor is a player (e.g., 'p0', 'p1', ...),
        or None if it's the dealer ('d') or other non-player actor.
        '''
        if self.actor.startswith('p'):
            return int(self.actor[1:]) - 1
        return None