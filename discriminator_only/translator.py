

from typing import List
from engine import Game, BOARD_SIZE, unitList

class ObservationEnum():
    def __init__(self, values: List):
        self.int_to_value = values
        self.value_to_int = {v: i for i, v in enumerate(values)}

    def decode(self, idx):
        return self.int_to_value[idx]

    def encode(self, value):
        return self.value_to_int[value]

HEXES = ObservationEnum([(i, j) for i in range BOARD_SIZE for j in range BOARD_SIZE])
LOCATIONS = ObservationEnum([(i, j) for i in range BOARD_SIZE for j in range BOARD_SIZE])

# 2x embeddings for unit types to encompass "mine" (True) and "opponent" (False)
UNIT_TYPES = ObservationEnum((u, c) for u in range(len(unitList)) for c in [True, False])

TERRAIN_TYPES = ObservationEnum(['none', 'water', 'graveyard'])

def translate(game: Game):
    hex_ids = [] # [num_hexes, 2 (location, terrain)]
    unit_ids = []  # [num_units,  2 (type (includes color), location)]
    for (i, j), hex in game.board.hexes():
        terrain = "graveyard" if hex.is_graveyard else "water" if hex.is_water else "none"
        hex_ids.append([
            HEXES.encode((i, j)),
            TERRAIN_TYPES.encode(terrain)
        ])
        
    for unit, (i, j) in game.units():
        unit_ids.append([
            UNIT_TYPES.encode((unit.index, unit.color == game.active_player_color)),
            LOCATIONS.encode((i, j))
            ])

    return {
        'hexes': hex_ids,
        'units': unit_ids,
    }
