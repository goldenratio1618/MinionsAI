
import enum
from typing import Tuple

from unit_type import UnitType


class ActionType(enum.Enum):
    MOVE = "move"
    SPAWN = "spawn"
    FINISH_PHASE = "finish"
    END_TURN = "end"

class Action():
    def __init__(self, action_type: ActionType):
        self.action_type: ActionType = action_type

class MoveAction(Action):
    """ 
    Can only be done during Move Phase
    Move unit from (xi, yi) to (xf, yf)
    Attack if the destination is occupied
    """
    def __init__(self, from_xy: Tuple[int, int], to_xy: Tuple[int, int]):
        super().__init__(ActionType.MOVE)
        self.from_xy = from_xy
        self.to_xy = to_xy

class SpawnAction(Action):
    """
    Can only be done during Spawn Phase
    Spawn unit of type unit_type at (x, y)
    """

    def __init__(self, unit_type: UnitType, to_xy: Tuple[int, int]):
        super().__init__(ActionType.SPAWN)
        self.unit_type = unit_type
        self.to_xy = to_xy

class FinishPhaseAction(Action):
    """
    Move on to the next phase (Move->Spawn->Turn_end)
    """
    def __init__(self):
        super().__init__(ActionType.FINISH_PHASE)

class EndTurnAction(Action):
    """
    Can only be done during Turn End Phase
    Either end the turn, or undo it and start over.
    """
    def __init__(self, undo_turn=False):
        super().__init__(ActionType.END_TURN)
        self.undo_turn = undo_turn
