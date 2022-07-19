

from functools import lru_cache
from typing import List

from ..action import AdvancePhaseAction, MoveAction, SpawnAction
from ..engine import Game, BOARD_SIZE, Phase
from ..unit_type import MAX_UNIT_HEALTH, ZOMBIE, unitList, MAX_SPEED_OR_RANGE
from collections import OrderedDict
import numpy as np

def convert_loc_to_pos(loc):
    return (loc // BOARD_SIZE, loc % BOARD_SIZE)

def convert_pos_to_loc(i, j):
    return i * BOARD_SIZE + j

class ObservationEnum():
    def __init__(self, values: List, none_value=False):
        if none_value:
            self.NULL = "null"
            values = [self.NULL] + values
        self.int_to_value = values

    def decode(self, idx):
        return self.int_to_value[idx]

    @lru_cache(maxsize=None)
    def encode(self, value):
        return self.int_to_value.index(value)

class Translator():
    HEXES = ObservationEnum([(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)])

    # 2x embeddings for unit types to encompass "mine" (True) and "opponent" (False)
    UNIT_TYPES_DISCRIMINATOR =  ObservationEnum([(u, c) for u in [u.name for u in unitList] for c in [True, False]], none_value=True)
    UNIT_TYPES_GENERATOR = ObservationEnum([(u, c, m, a, h) for u in [u.name for u in unitList] for c in [True, False] for m in [True, False] for a in [True, False] for h in range(1,MAX_UNIT_HEALTH+1)], none_value=True)
    TERRAIN_TYPES = ObservationEnum(['none', 'water', 'graveyard'])

    MAX_REMAINING_TURNS = 20
    MAX_MONEY = 20
    MAX_SCORE_DIFF = 20

    def __init__(self, mode):
        self.mode = mode
        self.num_things = BOARD_SIZE ** 2 + 4
        if self.mode == "generator":
            self.num_things += 1
            self.UNIT_TYPES = self.UNIT_TYPES_GENERATOR
        elif self.mode == "discriminator":
            self.UNIT_TYPES = self.UNIT_TYPES_DISCRIMINATOR
        else:
            raise ValueError("Mode must be 'discriminator' or 'generator'.")
        self.spawn_pos = BOARD_SIZE ** 2
        self.possibly_legal_actions = self.get_possible_legal_actions()
        self.untranslated_actions = self.untranslate_all_actions()

    def translate(self, game: Game):
        board_obs = [] # [num_hexes, 3 (location, terrain, unit_type)]
        for (i, j), hex in game.board.hexes():
            terrain = "graveyard" if hex.is_graveyard else "water" if hex.is_water else "none"
            if hex.unit is None:
                unit_type = self.UNIT_TYPES.NULL
            elif self.mode == "discriminator":
                unit_type = (hex.unit.type.name, hex.unit.color == game.active_player_color)
            elif self.mode == "generator":
                unit_type = (hex.unit.type.name, hex.unit.color == game.active_player_color, hex.unit.hasMoved, hex.unit.remainingAttack == 0, hex.unit.curr_health)

            board_obs.append([
                self.HEXES.encode((i, j)),
                self.TERRAIN_TYPES.encode(terrain),
                self.UNIT_TYPES.encode(unit_type)
            ])
        
        phase = (game.phase == Phase.MOVE)
        remaining_turns = game.remaining_turns
        all_money = game.money
        scores = game.get_scores
        # Clip the obs to be within bounds
        remaining_turns = min(remaining_turns, self.MAX_REMAINING_TURNS)
        
        money = min(all_money[game.active_player_color], self.MAX_MONEY)
        opp_money = min(all_money[1 - game.active_player_color], self.MAX_MONEY)
        score_diff = max(min(scores[game.active_player_color] - scores[game.inactive_player_color], self.MAX_SCORE_DIFF), -self.MAX_SCORE_DIFF) + self.MAX_SCORE_DIFF
        
        d = OrderedDict()
        d['board'] = np.array([board_obs])  # shape is [batch, num_items=BOARD_SIZE^2]
        d['money'] = np.array([[money]])
        d['remaining_turns'] = np.array([[remaining_turns]])  # shape is [batch, num_items=1]
        d['opp_money'] = np.array([[opp_money]])
        d['score_diff'] = np.array([[score_diff]])
        if self.mode == "generator":
            d['phase'] = np.array([[phase]])
            # legal_actions = [self.valid_actions(game).flatten()]
            # d['legal_actions'] = np.array([])
        return d

    def get_possible_legal_actions(self):
        possibly_legal = np.zeros((self.num_things, self.num_things), bool)
        for i in range(BOARD_SIZE ** 2 + 1):
            for j in range(BOARD_SIZE ** 2 + 1):
                # money -> board = spawn
                # money -> money = advance phase
                if i == self.spawn_pos:
                    possibly_legal[i,j] = True
                else:
                    if j == self.spawn_pos:
                        continue
                    x1, y1 = convert_loc_to_pos(i)
                    x2, y2 = convert_loc_to_pos(j)
                    if abs(x1-x2) <= MAX_SPEED_OR_RANGE and abs(y1-y2) <= MAX_SPEED_OR_RANGE and i != j:
                        possibly_legal[i,j] = True
        return possibly_legal

    def valid_actions(self, game: Game):
        legal = np.zeros((self.num_things, self.num_things), bool)
        if game.phase not in (Phase.MOVE, Phase.SPAWN):
            return legal
        for i in range(self.num_things):
            for j in range(self.num_things):
                if self.possibly_legal_actions[i,j]:
                    if i == self.spawn_pos:
                        # advance phase action
                        if j == self.spawn_pos:
                            legal[i,j] = True
                            continue
                        # wrong phase, action is illegal
                        if game.phase != Phase.SPAWN: 
                            continue
                        # spawn action
                        act = self.untranslate_action((i,j))
                        if game.process_single_spawn(act, False)[0]:
                            legal[i,j] = True
                    else:
                        # wrong phase, action is illegal
                        if game.phase != Phase.MOVE:
                            continue
                        # move action
                        act = self.untranslate_action((i,j))
                        if game.process_single_move(act, False)[0]:
                            legal[i,j] = True
                        # Don't allow swapping same units
                        from_unit = game.board.board[act.from_xy[0]][act.from_xy[1]].unit
                        to_unit = game.board.board[act.to_xy[0]][act.to_xy[1]].unit
                        if from_unit is not None and to_unit is not None and from_unit.type == to_unit.type:
                            legal[i,j] = False
        return legal

    def untranslate_all_actions(self):
        actions = [[None for i in range(self.num_things)] for j in range(self.num_things)]
        for i in range(self.num_things):
            for j in range(self.num_things):
                if i == self.spawn_pos:
                    # advance phase action
                    if j == self.spawn_pos:
                        actions[i][j] = AdvancePhaseAction()
                        continue
                    # spawn action
                    x2, y2 = convert_loc_to_pos(j)
                    act = SpawnAction(ZOMBIE, (x2,y2))
                    actions[i][j] = act
                else:
                    # move action
                    x1, y1 = convert_loc_to_pos(i)
                    x2, y2 = convert_loc_to_pos(j)
                    act = MoveAction((x1,y1), (x2,y2))
                    actions[i][j] = act
        return actions

    def untranslate_action(self, action):
        untranslated_action = self.untranslated_actions[action[0]][action[1]]
        assert not (untranslated_action is None)
        return untranslated_action

    @staticmethod
    def transpose_actions(actions: np.ndarray):
        # actions is shape [N, 2]
        # Where each of the 2 is an index into num_things;
        # i.e. 
        # flip is (i, j) -> (j, i)
        i, j = convert_loc_to_pos(actions)
        return np.where(actions < BOARD_SIZE ** 2, convert_pos_to_loc(j, i), actions)


    @staticmethod
    def rotate_actions(actions: np.ndarray):
        # rotate is (i, j) -> (4-i, 4-j)
        i, j = convert_loc_to_pos(actions)
        return np.where(actions < BOARD_SIZE ** 2, convert_pos_to_loc(4-i, 4-j), actions)

    @staticmethod
    def symmetries(obs, actions=None):
        # Takes an obs that came out of translator.translate()
        # and returns a list of symmetric observations
        # TODO(david) - this function really should be tested.
        board = obs['board']
        num = board.shape[0]
        board = np.reshape(board, [num, BOARD_SIZE, BOARD_SIZE, 3])
        all_boards = [board, np.transpose(board, axes=[0, 2, 1, 3])]
        rotated_boards = [np.flip(np.flip(b, axis=1), axis=2) for b in all_boards]
        all_boards = all_boards + rotated_boards
        all_boards = [np.reshape(b, [num, BOARD_SIZE ** 2, 3]) for b in all_boards]

        if actions is not None:
            assert actions.shape == (num, 2)
            transposed_actions = Translator.transpose_actions(actions)
            rotated_actions = Translator.rotate_actions(actions)
            rotated_flipped = Translator.rotate_actions(transposed_actions)
            all_actions = [actions, transposed_actions, rotated_actions, rotated_flipped]
        else:
            all_actions = [None] * 4
        

        new_obs = []
        for b in all_boards:
            d = OrderedDict()
            d['board'] = b # shape is [batch, num_items=BOARD_SIZE^2]
            d['money'] = obs['money']
            d['remaining_turns'] =  obs['remaining_turns'] 
            d['opp_money'] = obs['opp_money']
            d['score_diff'] =  obs['score_diff']
            if 'phase' in obs.keys():
                d['phase'] = obs['phase']
            new_obs.append(d)
        return new_obs, all_actions
