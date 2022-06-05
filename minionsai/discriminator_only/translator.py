

from typing import List
from ..engine import Game, BOARD_SIZE
from ..unit_type import unitList
import numpy as np

class ObservationEnum():
    def __init__(self, values: List, none_value=False):
        if none_value:
            self.NULL = "null"
            values = [self.NULL] + values
        self.int_to_value = values
        self.value_to_int = {v: i for i, v in enumerate(values)}

    def decode(self, idx):
        return self.int_to_value[idx]

    def encode(self, value):
        return self.value_to_int[value]

class Translator():
    HEXES = ObservationEnum([(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)])
    LOCATIONS = ObservationEnum([(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)])

    # 2x embeddings for unit types to encompass "mine" (True) and "opponent" (False)
    UNIT_TYPES =  ObservationEnum([(u, c) for u in [u.name for u in unitList] for c in [True, False]], none_value=True)

    TERRAIN_TYPES = ObservationEnum(['none', 'water', 'graveyard'])

    MAX_REMAINING_TURNS = 20
    MAX_MONEY = 20
    MAX_SCORE_DIFF = 20
    REMAINING_TURNS = ObservationEnum(list(range(MAX_REMAINING_TURNS + 1)))

    def translate(self, game: Game):
        board_obs = [] # [num_hexes, 3 (location, terrain, unit_type)]
        for (i, j), hex in game.board.hexes():
            terrain = "graveyard" if hex.is_graveyard else "water" if hex.is_water else "none"
            if hex.unit is None:
                unit_type = self.UNIT_TYPES.NULL
            else:
                unit_type = (hex.unit.type.name, hex.unit.color == game.active_player_color)
            
            board_obs.append([
                self.HEXES.encode((i, j)),
                self.TERRAIN_TYPES.encode(terrain),
                self.UNIT_TYPES.encode(unit_type)
            ])

        remaining_turns = game.remaining_turns
        all_money = game.money
        scores = game.get_scores
        # Clip the obs to be within bounds
        remaining_turns = min(remaining_turns, self.MAX_REMAINING_TURNS)
        
        money = min(all_money[game.active_player_color], self.MAX_MONEY)
        opp_money = min(all_money[1 - game.active_player_color], self.MAX_MONEY)
        score_diff = max(min(scores[game.active_player_color] - scores[game.inactive_player_color], self.MAX_SCORE_DIFF), -self.MAX_SCORE_DIFF) + self.MAX_SCORE_DIFF
        # TODO: Should the translator be in charge of calling ObservationEnum.encode()?
        return {
            'board': np.array([board_obs]),
            'remaining_turns': np.array([[remaining_turns]]),  # shape is [batch, num_items]
            'money': np.array([[money]]),
            'opp_money': np.array([[opp_money]]),
            'score_diff': np.array([[score_diff]])
        }
