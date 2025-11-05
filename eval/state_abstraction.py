import numpy as np
import duckdb
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List
from itertools import product
from pokerkit import HandHistory
from data.model.hand import Hand


@dataclass
class Feature:
    '''
    Defines how to bucket one feature of a hand/action pair.
    
    - name: feature name (string)
    - _fn: function(hand_history: HandHistory, action: Action) -> bucket label (str)
    - buckets: optional pre-defined list of possible bucket labels (for enumeration)
    '''
    name: str
    _fn: Callable[[HandHistory, int], str]
    buckets: List[str] = field(default_factory=list)


    def __call__(self, hh: HandHistory, a: int) -> str:
        '''
        Apply the bucket function to (hand, action). 
        Returns a string bucket label.
        '''
        return self._fn(hh, a)


@dataclass
class StateBucket:
    '''
    Represents a single combined state bucket.
    '''
    name: str
    features: Dict[str, str]


class Abstraction:
    '''
    Maps game states to discrete buckets.
    '''
    def __init__(self, features: List[Feature]):
        self.features = features
        self.feature_names = [f.name for f in features]
        self.buckets: Dict[str, StateBucket] = {}
    

    def __call__(self, hh: Hand, a: int) -> str:
        '''
        Map a (Hand, action_index) pair to a bucket key.
        '''
        feature_vals = {f.name: f(hh, a) for f in self.features}
        key = self.make_key(feature_vals, self.feature_names)
        # cache bucket
        if key not in self.buckets:
            self.buckets[key] = StateBucket(name=key, features=feature_vals)
        return key


    @staticmethod
    def make_key(feature_vals: Dict[str, str], order: List[str]) -> str:
        '''Canonical representation of a feature dict.'''
        return ' | '.join(f'({k}: {feature_vals[k]})' for k in order)


    def map_to_bucket(self, state: Dict[str, Any]) -> str:
        '''Given a raw state dict, produce or lookup a bucket key.'''
        feature_vals = {f.name: f.map(state.get(f.name)) for f in self.features}
        key = self.make_key(feature_vals, self.feature_names)
        if key not in self.buckets:
            self.buckets[key] = StateBucket(name=key, features=feature_vals)
        return key


    def all_possible_buckets(self) -> List[str]:
        '''Enumerate Cartesian product of all defined feature buckets.'''
        if any(not f.buckets for f in self.features):
            raise ValueError('All features must define .buckets to enumerate all combinations.')

        combos = product(*[f.buckets for f in self.features])
        return [
            self.make_key({f.name: val for f, val in zip(self.features, combo)}, self.feature_names)
            for combo in combos
        ]

# define a bucketing function
def bucket_position(hh: Hand, a: int):
    positions = {
        2: ['early', 'late'],
        3: ['early', 'early', 'late'],
        4: ['early', 'early', 'middle', 'late'],
        5: ['early', 'early', 'middle', 'middle', 'late'],
        6: ['early', 'early', 'early', 'middle', 'middle', 'late'],
        7: ['early', 'early', 'early', 'middle', 'middle', 'middle', 'late'],
        8: ['early', 'early', 'early', 'middle', 'middle', 'middle', 'late', 'late'],
        9: ['early', 'early', 'early', 'middle', 'middle', 'middle', 'late', 'late', 'late'],
        10: ['early', 'early', 'early', 'early', 'middle', 'middle', 'middle', 'late', 'late', 'late'],
        11: ['early', 'early', 'early', 'early', 'middle', 'middle', 'middle', 'middle', 'late', 'late', 'late'],
        12: ['early', 'early', 'early', 'early', 'early', 'middle', 'middle', 'middle', 'middle', 'late', 'late', 'late']
    }
    player_idx = hh.actions[a].player_idx
    return positions[hh.num_players][player_idx]

def bucket_street(hh: Hand, a: int):
    streets = {
        0: 'preflop',
        1: 'flop',
        2: 'turn',
        3: 'river'
    }
    street_index = hh.actions[a].street_index
    if street_index not in streets:
        return 'unknown'
    return streets[street_index]

def bucket_bet_size_bb(hh: Hand, a: int):
    action = hh.actions[a]
    if np.isnan(action.amount) or action.amount is None:
        return 'N/A'
    x = action.amount / hh.min_bet  # min_bet = 1bb
    if x <= 3: return '(0,3]'
    elif x <= 9: return '(3,9]'
    elif x <= 27: return '(9,27]'
    elif x <= 54: return '(27,54]'
    elif x <= 108: return '(54,108]'
    else: return '(108,inf)'

if __name__ == '__main__':
    f1 = Feature('position', bucket_position, ['early', 'middle', 'late'])
    f2 = Feature('street', bucket_street, ['preflop', 'flop', 'turn', 'river'])
    f3 = Feature('bet size', bucket_bet_size_bb, ['N/A', '(0, 3]', '(3, 9]', '(9, 27]', '(27, 54]', '(54, 108]', '(108, inf)'])
    s = Abstraction([f1, f2, f3])
    con = duckdb.connect('data/db/master.db')
    id: str = con.sql('select hand_id from hands order by num_actions desc limit 10;').fetchall()[4][0]
    hh = Hand.from_db(id, con)
    for action in hh.actions:
        if action.player_idx is not None and action.street_index is not None:
            print(f'action: {action.raw_action}')
            print(f'mapping to: {s(hh, action.action_index)}')
            print()