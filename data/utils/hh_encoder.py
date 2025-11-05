import numpy as np
from typing import List
from ..model.hand import Hand

CARD_TO_INDEX = {
    rank + suit: i
    for i, (rank, suit) in enumerate(
        [r + s for r in '23456789TJQKA' for s in 'shdc']
    )
}
NUM_CARD_FEATURES = 53  # 52 cards + 1 'unknown' slot
ACTION_TYPES = ['Check', 'Bet', 'Call', 'Fold', 'Raise']
ACTION_TO_INDEX = {a.lower(): i for i, a in enumerate(ACTION_TYPES)}


def encode_card(card: str) -> np.ndarray:
    '''One-hot encode a card string like 'As' or 'Td'. Handles unknown.'''
    vec = np.zeros(NUM_CARD_FEATURES, dtype=np.float32)
    if not card or card not in CARD_TO_INDEX:
        vec[-1] = 1.0  # Unknown
    else:
        vec[CARD_TO_INDEX[card]] = 1.0
    return vec


def encode_hole_cards(cards_str: str) -> np.ndarray:
    '''Encode two hole cards like 'AsKd'. Returns 106-dim (2 * 53).'''
    if not cards_str or len(cards_str) < 4:
        return np.concatenate([encode_card(None), encode_card(None)])
    c1, c2 = cards_str[:2], cards_str[2:]
    return np.concatenate([encode_card(c1), encode_card(c2)])



def encode_action_type(action_type: str) -> np.ndarray:
    vec = np.zeros(len(ACTION_TYPES), dtype=np.float32)
    key = (action_type or '').lower()
    if key in ACTION_TO_INDEX:
        vec[ACTION_TO_INDEX[key]] = 1.0
    return vec


def encode_street(street_index: int) -> np.ndarray:
    vec = np.zeros(4, dtype=np.float32)
    if 0 <= street_index <= 3:
        vec[street_index] = 1.0
    return vec


class HandEncoder:
    def __init__(self, max_players: int = 9):
        self.max_players = max_players

    def encode(self, hand: Hand) -> np.ndarray:
        '''Encode a full hand into a (num_actions, feature_dim) array.'''
        encoded_steps = []

        for action in hand.actions:
            player = next((p for p in hand.players if f'p{p.player_idx}' == action.actor), None)
            card_vec = encode_hole_cards(player.hole_cards if player else None)
            street_vec = encode_street(action.street_index)
            pos = (player.player_idx + 1) / hand.num_players if player else 0.0
            pot_bb = float(action.total_pot_amount) / float(hand.min_bet or 1.0)
            action_vec = encode_action_type(action.action_type)
            bet_bb = float(action.amount) / float(hand.min_bet) if not np.isnan(action.amount) else 0.0
            player_to_move = action.player_idx or -1.0

            x_t = np.concatenate([
                card_vec,               # 106
                street_vec,             # 4
                np.array([pos]),        # 1
                np.array([pot_bb]),     # 1
                action_vec,             # 5
                np.array([bet_bb]),     # 1
                np.array([player_to_move]),    # 1
            ])
            encoded_steps.append(x_t)

        return np.vstack(encoded_steps)
