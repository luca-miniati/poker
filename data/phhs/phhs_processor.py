import argparse
import re
import duckdb
import pandas as pd
import warnings
from pathlib import Path
from logging import Logger
from typing import Dict, List
from pokerkit import HandHistory
from data.utils.logs import setup_logging
from data.utils.hand_history import are_holdem_hole_cards_shown, ACTION_PARSING_REGEX


OUTPUT_DIR = Path('data/phhs/out')
LOG_DIR = OUTPUT_DIR / 'logs'
PARQUET_DIR = OUTPUT_DIR / 'parquet'
PARQUET_HANDS_DIR = PARQUET_DIR / 'hands'
PARQUET_PLAYERS_DIR = PARQUET_DIR / 'players'
PARQUET_ACTIONS_DIR = PARQUET_DIR / 'actions'
DB_DIR = Path('data/db')
DB_PATH = DB_DIR / 'master.db'

LOG_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_HANDS_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_PLAYERS_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_ACTIONS_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)


class PHHSProcessor:
    '''
    Processes poker hand histories in .phhs format.
    The processor goes through all .phhs files in `root_dir`, extracting
    data from the hand histories into 3 parquets:
    - `hands`
    - `players`
    - `actions`
    Then, 3 corresponding tables are created in `data/db/master.db`.
    '''
    def __init__(self, root_dir: str, batch_size: int = 10_000) -> None:
        self.root_dir: Path = Path(root_dir)
        self.log: Logger = setup_logging(log_dir=LOG_DIR, script_name=Path(__file__).name)
        self.batch_size: int = batch_size
        self.batch_index: int = 1
        self.hh_buffer: List[HandHistory] = []

    
    def parse_hand_from_hh(self, hh: HandHistory) -> Dict:
        '''
        Given a hand history, construct row for table `hands`.
        '''
        return {
            'hand_id': hh.hand,
            'variant': hh.variant,
            'bring_in': hh.bring_in,
            'small_bet': hh.small_bet,
            'big_bet': hh.big_bet,
            'min_bet': hh.min_bet,
            'num_players': len(hh.players),
            'num_actions': len(hh.actions),
            'venue': hh.venue,
            'table': hh.table,
        }

    
    def parse_players_from_hh(self, hh: HandHistory, hand_id: int) -> List[Dict]:
        '''
        Given a hand history, construct rows for table `players`.
        '''
        fields = [
            'hand_id',
            'ante',
            'blind_or_straddle',
            'name',
            'player_idx',
            'seat',
            'starting_stack',
            'finishing_stack',
            'amount_won'
        ]
        num_players = len(hh.players)
        return [
            dict(zip(fields, values))
            for values in zip(
                [hand_id] * num_players,
                hh.antes,
                hh.blinds_or_straddles,
                hh.players,
                range(1, num_players + 1),
                hh.seats,
                hh.starting_stacks,
                hh.finishing_stacks if hh.finishing_stacks
                                    else [None] * num_players,
                hh.winnings if hh.winnings
                            else [None] * num_players
            )
        ]

    
    def parse_actions_from_hh(self, hh: HandHistory, hand_id: int) -> List[Dict]:
        '''
        Given a hand history, construct rows for table `actions`.
        '''
        rows: List[Dict] = []
        for idx, action in enumerate(hh.actions):
            match = re.match(ACTION_PARSING_REGEX, action)
            if not match:
                self.log.error(f'Actions could not be parsed from hand history: {hh.actions}')
                rows.append({
                    'hand_id': hand_id,
                    'action_index': idx,
                    'actor': None,
                    'action_type': None,
                    'amount': None,
                    'cards': None,
                    'raw_action': action.strip()
                })
                continue

            actor = match.group('actor')
            action_type = match.group('action_type')
            args = match.group('args')
            amount = None
            cards = None

            if args:
                args = args.strip()
                if action_type == 'cbr':
                    try:
                        amount = float(args)
                    except ValueError:
                        amount = None
                elif action_type in {'dh', 'db', 'sm', 'sd'}:
                    cards = args

            rows.append({
                'hand_id': hand_id,
                'action_index': idx,
                'actor': actor,
                'action_type': action_type,
                'amount': amount,
                'cards': cards,
                'raw_action': action.strip()
            })
        return rows


    def flush_hh_buffer(self) -> None:
        '''
        Process all hand histories in `self.hh_buffer`, and clear the buffer.
        '''
        if not self.hh_buffer:
            return

        self.log.info('Flushing buffer')
        self.log.info(f'{len(self.hh_buffer)} hand histories in buffer')
        
        hands_rows: List[Dict] = []
        players_rows: List[Dict] = []
        actions_rows: List[Dict] = []
        for hh in self.hh_buffer:
            hand_row: Dict = self.parse_hand_from_hh(hh)
            hand_id: int = hand_row['hand_id']
            hands_rows.append(hand_row)
            players_rows.extend(self.parse_players_from_hh(hh, hand_id))
            actions_rows.extend(self.parse_actions_from_hh(hh, hand_id))

        self.log.info(f'{len(hands_rows)} rows extracted for hands')
        self.log.info(f'{len(players_rows)} rows extracted for players')
        self.log.info(f'{len(actions_rows)} rows extracted for actions')

        hands_df = pd.DataFrame(hands_rows)
        hands_df['hand_id'] = hands_df['hand_id'].astype(int)
        hands_df['num_players'] = hands_df['num_players'].astype(int)
        hands_df['num_actions'] = hands_df['num_actions'].astype(int)
        hands_df['bring_in'] = hands_df['bring_in'].astype('Float64')
        hands_df['small_bet'] = hands_df['small_bet'].astype('Float64')
        hands_df['big_bet'] = hands_df['big_bet'].astype('Float64')
        hands_df['min_bet'] = hands_df['min_bet'].astype(float)

        players_df = pd.DataFrame(players_rows)
        players_df['starting_stack'] = players_df['starting_stack'].astype('Float64')
        players_df['finishing_stack'] = players_df['finishing_stack'].astype('Float64')
        players_df['amount_won'] = players_df['amount_won'].astype('Float64')
        players_df['seat'] = players_df['seat'].astype(int)
        players_df['ante'] = players_df['ante'].astype(float)
        players_df['blind_or_straddle'] = players_df['blind_or_straddle'].astype(float)

        actions_df = pd.DataFrame(actions_rows)
        actions_df['amount'] = actions_df['amount'].astype(float)
        actions_df['action_index'] = actions_df['action_index'].astype(int)

        hands_filename = PARQUET_HANDS_DIR / f'part-{self.batch_index:04d}.parquet'
        players_filename = PARQUET_PLAYERS_DIR / f'part-{self.batch_index:04d}.parquet'
        actions_filename = PARQUET_ACTIONS_DIR / f'part-{self.batch_index:04d}.parquet'

        self.log.info(f'Exporting to {hands_filename}')
        hands_df.to_parquet(hands_filename, index=False)
        self.log.info(f'Exporting to {players_filename}')
        players_df.to_parquet(players_filename, index=False)
        self.log.info(f'Exporting to {actions_filename}')
        actions_df.to_parquet(actions_filename, index=False)

        self.hh_buffer.clear()
        self.batch_index += 1

    
    def is_valid_hand(self, hh: HandHistory) -> bool:
        '''
        Returns whether the given hand has all the information we want.
        '''
        if not are_holdem_hole_cards_shown(hh):
            return False
        
        if not hh.players:
            self.log.error(f'Invalid hand history (no player names): {hh}')
            return False
        
        if not hh.hand:
            self.log.error(f'Invalid hand history (no hand id): {hh}')
            return False
        
        if not hh.venue:
            self.log.error(f'Invalid hand history (no venue): {hh}')
            return False
        
        if not hh.table:
            self.log.error(f'Invalid hand history (no table): {hh}')
            return False
        
        if not hh.seats:
            self.log.error(f'Invalid hand history (no seats): {hh}')
            return False

        return True

    
    def process_file(self, file_path: Path) -> None:
        '''
        Process a single .phhs file.
        '''
        if file_path.suffix != '.phhs':
            self.log.error(f'Non-.phhs file found: {file_path}')
            return
        
        hhs: List[HandHistory] = []
        with open(file_path, 'rb') as f:
            hhs = list(HandHistory.load_all(f))

        for hh in hhs:
            if not self.is_valid_hand(hh):
                continue

            self.hh_buffer.append(hh)

            if len(self.hh_buffer) >= self.batch_size:
                self.flush_hh_buffer()


    def process_folder(self, folder_path: Path) -> None:
        '''
        Recursively process all folders, calling `self.process_file` on any
        files.
        '''
        self.log.info(f'Processing folder {folder_path.name}')
        for path in folder_path.iterdir():
            if path.is_dir():
                self.process_folder(path)
            elif path.is_file():
                self.process_file(path)
            else:
                self.log.error(f'Non-file/folder item found: {path}')


    def create_tables(self) -> None:
        self.log.info('Creating tables')
        with duckdb.connect(DB_PATH) as con:
            con.sql(f"create table hands as select * from '{PARQUET_HANDS_DIR}/*.parquet'")
            con.sql(f"create table players as select * from '{PARQUET_PLAYERS_DIR}/*.parquet'")
            con.sql(f"create table actions as select * from '{PARQUET_ACTIONS_DIR}/*.parquet'")

    def process(self) -> None:
        '''
        Process all folders in `self.root_dir`.
        '''
        warnings.filterwarnings('ignore', category=UserWarning, module='pokerkit.notation')
        folders: List[Path] = [p for p in self.root_dir.iterdir() if p.is_dir()]
        self.log.info(f'Found {len(folders)} folders')
        for folder_path in folders:
            self.process_folder(folder_path)
        
        self.flush_hh_buffer()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='PHHS Processor',
        description='Processes poker hand histories in .phhs format.',
    )
    parser.add_argument('root_dir', help='Path to the root directory containing .phhs files')

    args = parser.parse_args()
    root_dir = args.root_dir

    hhqp = PHHSProcessor(root_dir)
    hhqp.process()
    hhqp.create_tables()