

from typing import List
from engine import Game, BOARD_SIZE
from unit_type import unitList
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

HEXES = ObservationEnum([(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)])
LOCATIONS = ObservationEnum([(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)])

# 2x embeddings for unit types to encompass "mine" (True) and "opponent" (False)
UNIT_TYPES = ObservationEnum([(u, c) for u in range(len(unitList)) for c in [True, False]], none_value=True)

TERRAIN_TYPES = ObservationEnum(['none', 'water', 'graveyard'])

def translate(game: Game):
    board_obs = [] # [num_hexes, 3 (location, terrain, unit_type)]
    for (i, j), hex in game.board.hexes():
        terrain = "graveyard" if hex.is_graveyard else "water" if hex.is_water else "none"
        if hex.unit is None:
            unit_type = UNIT_TYPES.NULL
        else:
            unit_type = (hex.unit.index, hex.unit.color == game.active_player_color)
        
        board_obs.append([
            HEXES.encode((i, j)),
            TERRAIN_TYPES.encode(terrain),
            UNIT_TYPES.encode(unit_type)
        ])

    return {
        'board': np.array([board_obs]),
    }
