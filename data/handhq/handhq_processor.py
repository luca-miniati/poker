import glob
import logging
import os
import pandas as pd
from collections import defaultdict
from typing import Dict, List
from pokerkit import HandHistory


PROCESSED_DIR = 'data/handhq/processed'
OUTPUT_DIR = os.path.join(PROCESSED_DIR, 'player_to_phhs')
PLAYER_ID_MAPPING_PATH = os.path.join(PROCESSED_DIR, 'player_id_mapping.csv')

os.makedirs(os.path.dirname(PLAYER_ID_MAPPING_PATH), exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(PLAYER_ID_MAPPING_PATH):
    df = pd.DataFrame(columns=['player_str', 'player_id'])
    df.to_csv(PLAYER_ID_MAPPING_PATH, index=False)


class HandHQProcessor:
    '''
    Object to parse handhq hand histories.

    The processor goes through all or .phhs files in `root_dir`, maintaining a
    mapping of players to hand histories.

    At the end, it appends these hand histories to
    `data/handhq/processed/player_to_phhs/player_{player_id}.phhs`.

    Here, `player_id` is not the player identifier used in the .phh files.
    Rather, the player identifiers used in the .phh files are mapped to
    integer player id's in `data/handhq/processed/player_id_mapping.csv`.
    '''


    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.player_hhs_mapping: Dict[int, List[HandHistory]] = defaultdict(list)
        self.player_id_mapping = pd.read_csv(PLAYER_ID_MAPPING_PATH)

    
    def get_player_id(self, player_str):
        match = self.player_id_mapping[self.player_id_mapping['player_str'] == player_str]

        if match.empty:
            next_id = self.player_id_mapping['player_id'].max() + 1 if not self.player_id_mapping.empty else 1
            new_row = {'player_str': player_str, 'player_id': next_id}
            self.player_id_mapping = pd.concat([self.player_id_mapping, pd.DataFrame([new_row])], ignore_index=True)

            return next_id
        else:
            return match.iloc[0]['player_id']

    def parse_phhs_file(self, fn):
        hhs = []
        with open(fn, 'rb') as f:
            hhs = list(HandHistory.load_all(f))
        
        for hh in hhs:
            for player_str in hh.players:
                player_id = self.get_player_id(player_str)
                self.player_hhs_mapping[player_id].append(hh)


    def export_player_hhs_mapping(self):
        for player_id, hhs in self.player_hhs_mapping.items():
            fn = os.path.join(PROCESSED_DIR, 'player_to_phhs', f'player_{str(player_id)}.phhs')

            if os.path.exists(fn):
                with open(fn, 'rb') as f:
                    _hhs = list(HandHistory.load_all(f))
                    hhs.extend(_hhs)

            with open(fn, 'wb') as f:
                HandHistory.dump_all(hhs, f)


    def export_player_id_mapping(self):
        self.player_id_mapping.to_csv(PLAYER_ID_MAPPING_PATH, index=False)


    def process(self):
        fns = glob.glob(os.path.join(self.data_dir, '*'))

        for fn in fns:
            ext = os.path.splitext(fn)[1]
            if ext != '.phhs':
                logging.error(f'Non-phhs file found: {fn}')
                continue

            self.parse_phhs_file(fn)
        
        self.export_player_hhs_mapping()
        self.export_player_id_mapping()


if __name__ == '__main__':
    fp = HandHQProcessor('data/handhq/incoming')
    fp.process()