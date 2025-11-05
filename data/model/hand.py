import duckdb
from dataclasses import dataclass, field
from typing import List, Optional
from .player import Player
from .action import Action


@dataclass
class Hand:
    hand_id: str
    variant: str
    min_bet: float
    num_actions: int
    players: List[Player] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)

    @property
    def num_players(self) -> int:
        return len(self.players)

    def add_player(self, player: Player):
        self.players.append(player)

    def add_action(self, action: Action):
        self.actions.append(action)
        # Attach to player if possible
        player = next((p for p in self.players if f"p{p.player_idx}" == action.actor), None)
        if player:
            player.add_action(action)

    def get_player_by_name(self, name: str) -> Optional[Player]:
        return next((p for p in self.players if p.name == name), None)

    @staticmethod
    def from_db(hand_id: str, con: duckdb.DuckDBPyConnection) -> 'Hand':
        # Load hand-level info
        hand_row = con.sql(f"SELECT * FROM hands WHERE hand_id='{hand_id}'").df().iloc[0]
        hand = Hand(
            hand_id=hand_row['hand_id'],
            variant=hand_row['variant'],
            min_bet=hand_row['min_bet'],
            num_actions=hand_row['num_actions']
        )

        # Load players
        players_df = con.sql(f"SELECT * FROM players WHERE hand_id='{hand_id}'").df()
        for _, row in players_df.iterrows():
            player = Player(
                name=row['name'],
                hand_id=row['hand_id'],
                player_idx=row['player_idx'],
                blind_or_straddle=row.get('blind_or_straddle', 0),
                hole_cards=row.get('hole_cards'),
                payoff=row.get('payoff', 0)
            )
            hand.add_player(player)

        # Load actions
        actions_df = con.sql(f"SELECT * FROM actions WHERE hand_id='{hand_id}'").df()
        for _, row in actions_df.iterrows():
            action = Action(
                hand_id=row['hand_id'],
                actor=row['actor'],
                action_index=row['action_index'],
                action_type=row['action_type'],
                amount=row.get('amount', 0),
                total_pot_amount=row.get('total_pot_amount', 0),
                street_index=row.get('street_index', 0),
                cards=row.get('cards'),
                is_terminal=row.get('is_terminal', False),
                raw_action=row.get('raw_action')
            )
            hand.add_action(action)

        return hand