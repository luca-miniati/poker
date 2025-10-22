import duckdb
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List
from itertools import product
from pokerkit import HandHistory
from data.model.hand import Hand


@dataclass
class StateFeature:
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


class StateAbstraction:
    '''
    Maps game states to discrete buckets.
    '''
    def __init__(self, features: List[StateFeature]):
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
    player_idx = hh.actions[a - 1].player_idx
    return positions[hh.num_players][player_idx]

if __name__ == '__main__':

    f = StateFeature('position', bucket_position, ['early', 'middle', 'late'])
    s = StateAbstraction([f])
    con = duckdb.connect('data/db/master.db')
    hh = Hand.from_db('26505473230', con)
    for action in hh.actions:
        if action.player_idx is not None:
            print(f'action: {action.raw_action}')
            print(f'mapping to: {s(hh, action.action_index)}')