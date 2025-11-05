import re
import duckdb
import tomli_w
import pandas as pd
from typing import Optional, Tuple, List
from pokerkit import HandHistory

PREFLOP = 0
FLOP = 1
TURN = 2
RIVER = 3
HOLDEM_CARD_REGEX = r'([2-9TJQKA][cdhs])' 
HOLDEM_HOLE_CARDS_SHOWN_REGEX = r'p(\d+)\s+sm\s+' + 2 * HOLDEM_CARD_REGEX
DEAL_FLOP_REGEX = r'd db ' + 3 * HOLDEM_CARD_REGEX
DEAL_TURN_REGEX = r'd db ' + HOLDEM_CARD_REGEX
DEAL_RIVER_REGEX = r'd db ' + HOLDEM_CARD_REGEX
# action parsing regex:
# ^\s*                      # optional leading whitespace
# (?P<actor>\S+)            # actor: p1, p2, d
# \s+
# (?P<action_type>\S+)      # action type: f, cc, cbr, dh, db, sm, etc.
# (?:\s+(?P<args>[^#]+))?   # optional arguments (cards, amount), stop at #
# (?:\s*\#.*)?              # optional commentary
# \s*$
ACTION_PARSING_REGEX = r'^\s*(?P<actor>\S+)\s+(?P<action_type>\S+)(?:\s+(?P<args>[^#]+))?(?:\s*\#.*)?\s*$'


def get_holdem_hole_cards(action: str) -> Optional[Tuple[str]]:
    '''
    Returns a tuple of cards, if the action shows hole cards. Otherwise
    returns None.
    '''
    match = re.match(HOLDEM_HOLE_CARDS_SHOWN_REGEX, action)
    if not match:
        return None
    return match.group(2), match.group(3)


def are_holdem_hole_cards_shown(hh: HandHistory) -> bool:
    '''
    Returns whether hole cards are shown at some point in the hand.
    '''
    match = re.search(HOLDEM_HOLE_CARDS_SHOWN_REGEX, ''.join(hh.actions))
    return match is not None


def load_hhs_from_db(hand_ids: List[str], con: duckdb.DuckDBPyConnection) -> List[HandHistory]:
    '''
    Given a list of hand ids, query the database and return a list of `HandHistory`s.
    '''
    if not hand_ids:
        return []


    hand_ids_str = ','.join(map(lambda x: "'" + x + "'", hand_ids))
    df_hands = con.execute(f'SELECT * FROM hands WHERE hand_id IN ({hand_ids_str})').fetchdf()
    df_players = con.execute(f'SELECT * FROM players WHERE hand_id IN ({hand_ids_str}) ORDER BY player_idx').fetchdf()
    df_actions = con.execute(f'SELECT * FROM actions WHERE hand_id IN ({hand_ids_str}) ORDER BY action_index').fetchdf()

    players_by_hand = df_players.groupby('hand_id')
    actions_by_hand = df_actions.groupby('hand_id')

    hand_histories = []
    for _, hand_row in df_hands.iterrows():
        hand_id = hand_row['hand_id']

        player_rows = players_by_hand.get_group(hand_id)
        action_rows = actions_by_hand.get_group(hand_id)

        def clean_dict(d):
            result = {}
            for k, v in d.items():
                if isinstance(v, list) or pd.notna(v):
                    result[k] = v
            return result

        hand_dict = clean_dict({
            'variant': hand_row.get('variant'),
            'antes': [p['ante'] for _, p in player_rows.iterrows()],
            'blinds_or_straddles': [p['blind_or_straddle'] for _, p in player_rows.iterrows()],
            'min_bet': hand_row.get('min_bet'),
            'players': list(player_rows['name']),
            'starting_stacks': list(player_rows['starting_stack']),
            'actions': list(action_rows['raw_action']),
            'venue': hand_row.get('venue'),
            'table': hand_row.get('table'),
        })

        phh_string = tomli_w.dumps(hand_dict)
        hh = HandHistory.loads(phh_string)
        hand_histories.append(hh)

    return hand_histories