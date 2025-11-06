import numpy as np
from typing import List
from ..model.hand import Hand, Player, Action

RANKS = '23456789TJQKA'
SUITS = 'cdhs'
CARD_TO_INDEX = {
    rank + suit: i
    for i, (rank, suit) in enumerate(
        [r + s for r in '23456789TJQKA' for s in 'shdc']
    )
}
NUM_CARD_FEATURES = 52
ACTION_TYPES = ['Check', 'Bet', 'Call', 'Fold', 'Raise']
ACTION_TO_INDEX = {a.lower(): i for i, a in enumerate(ACTION_TYPES)}


def encode_card_one_hot(card: str) -> np.ndarray:
    '''One-hot encode a card string like 'As' or 'Td'. Handles (??) as a vector of 0's.'''
    vec = np.zeros(NUM_CARD_FEATURES, dtype=np.float32)
    if card is not None and card in CARD_TO_INDEX:
        vec[CARD_TO_INDEX[card]] = 1.0
    return vec


def encode_card_indexed(card: str) -> np.ndarray:
    '''Encodes a card string with 2 numbers: rank index and suit index.'''
    vec = np.zeros(2, dtype=np.float32)
    if card is not None:
        rank, suit = card[0], card[1]
        if rank in RANKS:
            rank_idx = RANKS.index(rank)
            vec[0] = float(rank_idx)
        if suit in SUITS:
            suit_idx = SUITS.index(suit)
            vec[1] = float(suit_idx)
    return vec


def encode_hole_cards_one_hot(cards: List[str]) -> np.ndarray:
    '''Encode two hole cards like ['As', 'Kd']. Returns 104-dim (2 * 52).'''
    if not cards:
        return np.concatenate([encode_card_one_hot(None), encode_card_one_hot(None)])
    return np.concatenate([encode_card_one_hot(cards[0]), encode_card_one_hot(cards[1])])


def encode_hole_cards_indexed(cards: List[str]) -> np.ndarray:
    '''Encode two hole cards like ['As', 'Kd']. Returns a vector of 4 numbers.'''
    if not cards:
        return np.concatenate([encode_card_indexed(None), encode_card_indexed(None)])
    return np.concatenate([encode_card_indexed(cards[0]), encode_card_indexed(cards[1])])


def encode_community_cards_one_hot(cards: List[str]) -> np.ndarray:
    '''Encode 3 cards like ['As', 'Kd', 'Ah']. Returns 156-dim (3 * 52).'''
    return np.concatenate([
        encode_card_one_hot(cards[0] if len(cards) >= 1 else None),
        encode_card_one_hot(cards[1] if len(cards) >= 2 else None),
        encode_card_one_hot(cards[2] if len(cards) >= 3 else None),
    ])


def encode_cards_indexed(cards: List[str]) -> np.ndarray:
    '''Encode 3 cards like ['As', 'Kd', 'Ah']. Returns a vector of 6 numbers.'''
    return np.concatenate([
        encode_card_indexed(cards[0] if len(cards) >= 1 else None),
        encode_card_indexed(cards[1] if len(cards) >= 2 else None),
        encode_card_indexed(cards[2] if len(cards) >= 3 else None),
    ])


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
    output_dim = 19

    def __init__(self, max_players: int = 9):
        self.max_players = max_players

    def encode(self, hand: Hand) -> np.ndarray:
        '''Encode a full hand into a (num_actions, feature_dim) array.'''
        encoded_steps = []

        for action in hand.actions:
            player: Player = next((p for p in hand.players if f'p{p.player_idx}' == action.actor), None)
            # represents cards dealt to player or board. action_type_vec stores which one it is
            card_vec = encode_cards_indexed(action.cards)
            street_vec = encode_street(action.street_index)
            pos = (player.player_idx + 1) / hand.num_players if player else 0.0
            pot_bb = float(action.total_pot_amount) / float(hand.min_bet or 1.0)
            action_type_vec = encode_action_type(action.action_type)
            bet_bb = float(action.amount) / float(hand.min_bet) if not np.isnan(action.amount) else 0.0
            player_to_move = action.player_idx or -1.0

            x_t = np.concatenate([
                card_vec,
                street_vec,
                np.array([pos]),
                np.array([pot_bb]),
                action_type_vec,
                np.array([bet_bb]),
                np.array([player_to_move]),
            ])
            encoded_steps.append(x_t)

        return np.vstack(encoded_steps)
