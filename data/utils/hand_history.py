import re
from typing import Optional, Tuple
from pokerkit import HandHistory


HOLDEM_HOLE_CARDS_SHOWN_REGEX = r'p(\d+)\s+sm\s+([2-9TJQKA][cdhs])([2-9TJQKA][cdhs])'
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