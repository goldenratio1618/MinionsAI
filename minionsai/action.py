
import enum
from typing import List, Tuple

from .unit_type import UnitType


class ActionType(enum.Enum):
    MOVE = "move"
    SPAWN = "spawn"
    ADVANCE_PHASE = "advance_phase"

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

    def __repr__(self):
        return f"<MoveAction {self.from_xy} -> {self.to_xy}>"

class SpawnAction(Action):
    """
    Can only be done during Spawn Phase
    Spawn unit of type unit_type at (x, y)
    """

    def __init__(self, unit_type: UnitType, to_xy: Tuple[int, int]):
        super().__init__(ActionType.SPAWN)
        self.unit_type = unit_type
        self.to_xy = to_xy

    def __repr__(self):
        return f"<SpawnAction {self.unit_type.name} @ {self.to_xy}>"

class AdvancePhaseAction(Action):
    """
    Advance phase by one step. (Move phase to spawn phase, or spawn phase to end turn.)
    """
    def __init__(self):
        super().__init__(ActionType.ADVANCE_PHASE)

    def __repr__(self):
        return "Advance phase."

class ActionList():
    def __init__(self, move_phase: List[Action], spawn_phase: List[Action]):
        self.move_phase: List[Action] = move_phase
        self.spawn_phase: List[Action] = spawn_phase

    def __repr__(self):
        return f"<ActionList Move Phase: {self.move_phase} ||||| Spawn Phase: {self.spawn_phase}>"

    @staticmethod
    def from_single_list(actions):
        # TODO return ActionList from a list of single actions
        return ActionList(actions, [])