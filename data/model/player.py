from dataclasses import dataclass, field
from typing import List, Optional
from .action import Action

@dataclass
class Player:
    name: str
    hand_id: str
    player_idx: int
    blind_or_straddle: float = 0
    hole_cards: Optional[str] = None
    payoff: float = 0
    actions: List[Action] = field(default_factory=list)

    def add_action(self, action: Action):
        self.actions.append(action)