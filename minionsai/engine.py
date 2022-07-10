from collections import defaultdict
import enum
from functools import lru_cache
import random
from typing import Iterator, Optional, Tuple, List
from .unit_type import UnitType, NECROMANCER, ZOMBIE, unit_type_from_name
from .action import ActionType, Action, ActionList, MoveAction, SpawnAction
import numpy as np

BOARD_SIZE = 5

# distance function between two hexes
def dist(xi, yi, xf, yf):
    return max(abs(yf - yi), abs(xf - xi) + (abs(yf - yi) if (xi > xf) == (yi > yf) else 0))

# return a tuple of tuples of adjacent hexes
# Important to return a tuple rather than a list, so that it's immutable
# Otherwise scary stuff might happen when we cache it
@lru_cache()
def adjacent_hexes(x, y) -> Tuple:
    hex_list = []
    if x > 0: hex_list.append((x-1,y))
    if x < (BOARD_SIZE - 1): hex_list.append((x+1,y))
    if y > 0:
        hex_list.append((x,y-1))
        if x < (BOARD_SIZE - 1): hex_list.append((x+1,y-1))
    if y < (BOARD_SIZE - 1):
        hex_list.append((x,y+1))
        if x > 0: hex_list.append((x-1,y+1))
    return tuple(hex_list)

class Board():
    def __init__(self, water_locs, graveyard_locs):
        self.board = [[Hex((i,j) in water_locs, (i,j) in graveyard_locs) for j in range(BOARD_SIZE)] for i in range(BOARD_SIZE)]

    def copy(self) -> "Board":
        """
        Returns a copy of the board. 
        Temporary Unit properties (damage, etc) are NOT copied; only use this at turn start.
        """
        b = Board([], [])
        for (i, j), hex in self.hexes():
            b.board[i][j].is_water = hex.is_water
            b.board[i][j].is_graveyard = hex.is_graveyard
            if hex.unit is not None:
                copied_unit = hex.unit.copy()
                b.board[i][j].add_unit(copied_unit)
        return b

    def board_properties(self):
        return [(i, j, self.board[i][j].is_water, self.board[i][j].is_graveyard) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)]

    def print_board_properties(self):
        for entry in self.board_properties():
            print(*entry)

    def hex_state(self, i, j):
        if self.board[i][j].unit is None:
            unit_index = None
            unit_color = None
        else:
            unit_index = self.board[i][j].unit.type.name[0]
            unit_color = self.board[i][j].unit.color
        return i, j, unit_index, unit_color

    def board_state(self):
        return [self.hex_state(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)]

    def print_board_state(self):
        for entry in self.board_state():
            print(*entry)

    def hexes(self):
        # use like `for (i, j), hex in self.hexes():`
        for i, row in enumerate(self.board):
            for j, hex in enumerate(row):
                yield (i, j), hex

    def encode_json(self):
        return [[{
            'is_water': hex.is_water,
            'is_graveyard': hex.is_graveyard,
            'unit': hex.unit.encode_json() if hex.unit is not None else None
        } for hex in row] for row in self.board]

    @staticmethod
    def decode_json(json_data):
        board = Board([], [])
        for i, row in enumerate(json_data):
            for j, hex in enumerate(row):
                board.board[i][j].is_water = hex['is_water']
                board.board[i][j].is_graveyard = hex['is_graveyard']
                if hex['unit'] is not None:
                    board.board[i][j].add_unit(Unit.decode_json(hex['unit']))
        return board
                
class Hex():
    def __init__(self, is_water, is_graveyard):
        self.is_water = is_water
        self.is_graveyard = is_graveyard
        self.terrain = 0
        self.unit = None

    def add_unit(self, unit):
        self.unit = unit
    
    def remove_unit(self):
        self.unit = None

# This multiple inheritance is a bit horrifying,
# but according to SO it's the way to get enum's to be json-able.
class Phase(str, enum.Enum):
    MOVE = "move"  # Move phase
    SPAWN = "spawn"  # Spawn Phase
    TURN_END = "turn_end"  # After spawn phase, but haven't yet run next_turn()
    GAME_OVER = "game_over"  # Game is done.

    def __eq__(self, __x: object) -> bool:
        return self.value == __x.value

    def __hash__(self) -> int:
        return hash(self.value)

class Game():
    def __init__(self, 
                 money=(2, 4),
                 board=None, 
                 income_bonus=1,
                 new_game=True,
                 active_player_color=0,
                 max_turns=20, 
                 symmetrize=True,
                 min_graveyards=3,
                 max_graveyards=8,
                 phase=Phase.TURN_END,
                 record_metrics=True):
        """
        Important API pieces:
            game.full_turn(action_list) - process an action list for the current player
            game.next_turn() - advance to the next player's turn. Should be called once before each call to game.full_turn.

        Instead of doing full_turn, you can do it bit by bit yourself:
            game.next_turn()
            if game.done:
                ...

            # phase = Phase.MOVE
            game.process_single_action(action1)
            game.process_single_action(action2)
            ...
            game.end_move_phase()
            # phase = Phase.SPAWN
            game.process_single_action(action3)
            game.process_single_action(action4)
            ...
            game.end_spawn_phase()
            # phase = Phase.TURN_END
        """
        if board is None:
            # starting position: captains on opposite corners with one graveyard in center
            new_graveyards = []
            while (len(new_graveyards) < min_graveyards or len(new_graveyards) > max_graveyards):
                graveyard_locs = [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if random.random() < 0.25 and 1 <= i + j and i + j <= (2 * BOARD_SIZE - 3)]
                # symmetrize: remove graveyards with 50% probability and otherwise mirror reflect them
                if symmetrize:
                    new_graveyards = []
                    for loc in graveyard_locs:
                        i, j = loc
                        if (2 * i == BOARD_SIZE - 1 and 2 * j == BOARD_SIZE - 1):
                            new_graveyards.append(loc)
                        else:
                            if random.random() < 0.5:
                                new_graveyards.append(loc)
                                new_graveyards.append((BOARD_SIZE - 1 - i, BOARD_SIZE - 1 - j))
                else:
                    new_graveyards = graveyard_locs
            water_locs = []
            board = Board(water_locs, new_graveyards)
            board.board[0][0].add_unit(Unit(0, NECROMANCER)) # yellow captain
            board.board[BOARD_SIZE - 1][ BOARD_SIZE - 1].add_unit(Unit(1, NECROMANCER)) # blue captain
        self.board: Board = board

        self.income_bonus: int = income_bonus

        # money for two sides
        self.money = list(money)
        self.active_player_color: int = active_player_color
        self.phase: Phase = phase

        self.remaining_turns: int = max_turns

        if new_game:
            # Because we are going to call "next_turn()" before the start of the first turn,
            # The game really starts at the end of turn negative 1.
            self.remaining_turns += 1
            self.active_player_color = 1 - self.active_player_color
            assert self.phase == Phase.TURN_END, "New game should start with TURN_END phase."

        self.record_metrics = record_metrics
        if self.record_metrics:
            self._metrics = (defaultdict(float), defaultdict(float))

    @property
    def done(self) -> bool:
        return self.remaining_turns <= 0

    @property
    def get_scores(self):
        """
        Returns scores of both players.
        """
        scores = self.money.copy()
        for unit, _ in self.units_with_locations():
            if unit.type != NECROMANCER:
                scores[unit.color] += unit.type.cost
        return scores
        
    def __hash__(self):
        board_tuple = tuple((
            self.board.board[i][j].is_water, 
            self.board.board[i][j].is_graveyard,
            None if self.board.board[i][j].unit is None else hash(self.board.board[i][j].unit),
            )
            for i in range(BOARD_SIZE) for j in range(BOARD_SIZE))
        return hash((board_tuple, tuple(self.money), self.active_player_color, self.phase, self.remaining_turns))

    @property
    def winner(self) -> int:
        """
        Returns index of winning player (0 or 1)
        """
        assert self.done
        return np.argmax(self.get_scores)

    @property
    def inactive_player_color(self) -> int:
        return 1 - self.active_player_color

    def get_metrics(self, color):
        if self.record_metrics:
            return self._metrics[color]
        else:
            raise ValueError("Metrics are not being recorded.")

    def add_to_metric(self, color, key, amount):
        if self.record_metrics:
            self._metrics[color][key] += amount

    def pretty_print(self, do_print=True):
        """
        Prints board in ascii
        """
        rectangle_grid_dim = BOARD_SIZE * 2 - 1
        rectangle_grid = np.array([[" " for _ in range(rectangle_grid_dim)] for _ in range(rectangle_grid_dim)], dtype=str)
        for (i, j), hex in self.board.hexes():
            grid_y, grid_x = (i-j + (rectangle_grid_dim - 1) // 2), i+j
            if hex.unit is None:
                if hex.is_water:
                    char = '~'
                elif hex.is_graveyard:
                    char = '$'
                else:
                    char = '-'
            else:
                char = hex.unit.type.name[0]
                if hex.unit.color == 0:
                    char = char.lower()
                else:
                    char = char.upper()
                # TODO units on graveyards / water?

            rectangle_grid[grid_x, grid_y] = char
        row_strs = ["".join(row) for row in rectangle_grid]

        # Add a margin for displaying extra info on the side
        row_strs = [row + "  " for row in row_strs]
        row_strs[0] += f"${self.money[0]}"
        row_strs[-1] += f"${self.money[1]}"
        phase = self.phase.name.capitalize()
        if self.active_player_color == 0:
            row_strs[1] += phase
        else:
            row_strs[-2] += phase
        result = "\n".join(row_strs)
        if do_print: print(result)
        return result

    def units_with_locations(self, color=None) -> Iterator[Tuple["Unit", Tuple[int, int]]]:
        for (i, j), hex in self.board.hexes():
            if hex.unit is not None and (color is None or hex.unit.color == color):
                yield (hex.unit, (i, j))

    def next_turn(self) -> None:
        assert self.phase == Phase.TURN_END, f"Can only call game.next_turn() from TURN_END phase; received call in {self.phase.name}. Did you forget to call game.end_spawn_phase()?"
        self.active_player_color = self.inactive_player_color
        self.remaining_turns -= 1
        if self.done:
            self.phase = Phase.GAME_OVER
            for color in (0, 1):
                self.add_to_metric(color, 'final_money', self.money[color])
                self.add_to_metric(color, 'final_num_units', len(list(self.units_with_locations(color=color))))
            self.add_to_metric(self.winner, 'wins', 1)
            return

        self.phase = Phase.MOVE

    def process_single_action(self, action: Action) -> Tuple[bool, Optional[str]]:
        if self.phase == Phase.MOVE:
            # Note: We need to compare the names not the enum types,
            # in case the agent submitted an action from their local copy of the codebase
            # which is technically a different enum type.
            if action.action_type.name == ActionType.MOVE.name:
                return self.process_single_move(action)
            elif action.action_type.name == ActionType.ADVANCE_PHASE.name:
                self.end_move_phase()
                return True, None
            else:
                raise ValueError(f"Wrong action type ({action.action_type}) for Move Phase. Did you forget to call game.end_spawn_phase() before moving on to the spawn_phase?")
        elif self.phase == Phase.SPAWN:
            if action.action_type.name == ActionType.SPAWN.name:
                return self.process_single_spawn(action)
            elif action.action_type.name == ActionType.ADVANCE_PHASE.name:
                self.end_spawn_phase()
                return True, None
            else:
                raise ValueError(f"Wrong action type ({action.action_type}) for Spawn Phase.")
        elif self.phase == Phase.TURN_END:
            if action.action_type.name == ActionType.ADVANCE_PHASE.name:
                return True, None
            else:
                raise ValueError("Tried to process actions during TURN_END phase. Did you forget to call game.next_turn() before the next player's actions?")
        else:
            raise ValueError(f"Wrong phase ({self.phase}) for processing actions.")

    def process_single_move(self, move_action: MoveAction, modify=True) -> Tuple[bool, Optional[str]]:
        # returns true if move is legal & succesful, false otherwise
        # Second entry is the error message if move is illegal

        assert self.phase == Phase.MOVE, f"Tried to move during phase {self.phase}"
        # TODO: Make sure there is a path from origin to destination
        xi, yi = move_action.from_xy
        xf, yf = move_action.to_xy
        # make sure from hex has unit that can move
        if self.board.board[xi][yi].unit == None: return False, "No unit at start location"
        # only move your own units
        if self.board.board[xi][yi].unit.color != self.active_player_color: return False, "Unit at start location is not your unit"
        # make sure origin and destination are sufficiently close
        speed = self.board.board[xi][yi].unit.type.speed
        attack_range = self.board.board[xi][yi].unit.type.attack_range
        distance = dist(xi, yi, xf, yf)
        # if target hex is empty parse move as movement
        if self.board.board[xf][yf].unit == None:
            if distance > speed: return False, f"Move is too far ({distance} > {speed})"
            if self.board.board[xi][yi].unit.hasMoved: return False, "Unit has already moved"
            if not self.board.board[xi][yi].unit.type.flying and self.board.board[xf][yf].is_water: return False, "Unit is not flying and is trying to move to water"
            if not modify:
                return True, None
            self.board.board[xi][yi].unit.hasMoved = True
            if self.board.board[xi][yi].unit.type.lumbering:
                self.board.board[xi][yi].unit.remainingAttack = 0
            self.board.board[xf][yf].add_unit(self.board.board[xi][yi].unit)
            self.board.board[xi][yi].remove_unit()
        # if target hex is occupied by friendly unit then swap the units
        elif self.board.board[xf][yf].unit.color == self.active_player_color:
            if distance > speed: return False, f"Move is too far ({distance} > {speed})"
            if distance > self.board.board[xf][yf].unit.type.speed: return False, f"Move is too far for swapping unit ({distance} > {self.board.board[xf][yf].unit.type.speed})"
            if self.board.board[xi][yi].unit.hasMoved: return False, "Unit has already moved"
            if self.board.board[xf][yf].unit.hasMoved: return False, "Swapping unit has already moved"
            if not self.board.board[xi][yi].unit.type.flying and self.board.board[xf][yf].is_water: return False, "Unit is not flying and is trying to move to water"
            if not self.board.board[xf][yf].unit.type.flying and self.board.board[xi][yi].is_water: return False, "Swapping unit is not flying and is trying to move to water"
            if not modify:
                return True, None
            self.board.board[xi][yi].unit.hasMoved = True
            self.board.board[xf][yf].unit.hasMoved = True
            if self.board.board[xi][yi].unit.type.lumbering:
                self.board.board[xi][yi].unit.remainingAttack = 0
            if self.board.board[xf][yf].unit.type.lumbering:
                self.board.board[xf][yf].unit.remainingAttack = 0
            temp = self.board.board[xi][yi].unit
            self.board.board[xi][yi].remove_unit()
            self.board.board[xi][yi].add_unit(self.board.board[xf][yf].unit)
            self.board.board[xf][yf].remove_unit()
            self.board.board[xf][yf].add_unit(temp)
        # if target hex is occupied by enemy unit, then attack
        elif self.board.board[xf][yf].unit.color != self.active_player_color:
            if self.board.board[xi][yi].unit.remainingAttack == 0: return False, "Unit has no remaining attack"
            if distance > attack_range: return False, f"Attack is too far ({distance} > {attack_range})"
            # attacking prevents later movement
            if not modify:
                return True, None
            self.board.board[xi][yi].unit.hasMoved = True
            # unsummon removes non-persistent unit from board and refunds cost
            if self.board.board[xi][yi].unit.type.unsummoner and not self.board.board[xf][yf].unit.type.persistent:
                self.money[self.inactive_player_color] += self.board.board[xf][yf].unit.type.cost
                self.board.board[xf][yf].remove_unit()
                self.board.board[xi][yi].unit.remainingAttack = 0
                self.add_to_metric(self.active_player_color, "bounces", 1)
                return True, None
            # flurry deals 1 attack
            if self.board.board[xi][yi].unit.type.flurry:
                self.board.board[xi][yi].unit.remainingAttack -= 1
                attack_outcome = self.board.board[xf][yf].unit.receive_attack(1)
            # otherwise deal full attack
            else:
                self.board.board[xi][yi].unit.remainingAttack = 0
                attack_outcome = self.board.board[xf][yf].unit.receive_attack(self.board.board[xi][yi].unit.type.attack)
            # process dead unit, if applicable
            if attack_outcome >= 0:
                # remove unit from board
                self.board.board[xf][yf].remove_unit()
                # process rebate
                self.money[self.inactive_player_color] += attack_outcome
                self.add_to_metric(self.active_player_color, "kills", 1)
        return True, None

    def process_single_spawn(self, spawn_action: SpawnAction, modify=True) -> Tuple[bool, Optional[str]]:
        # returns true if spawn is legal & succesful, false otherwise

        assert self.phase == Phase.SPAWN, f"Tried to spawn during phase {self.phase}"
        # TODO: treat reinforcements explicitly (so that one can spawn bounced units without paying dollars)
        unit_type: UnitType = spawn_action.unit_type
        x, y = spawn_action.to_xy
        # check to see if we have enough money
        cost = unit_type.cost
        if cost > self.money[self.active_player_color]: return False, f"Not enough money to spawn {unit_type.name}"
        # check to make sure hex is unoccupied
        if self.board.board[x][y].unit != None: return False, f"Hex {x}, {y} is already occupied"
        # check to make sure we are adjacent to spawner
        adjacent_spawner = False
        for square in adjacent_hexes(x, y):
            ax, ay = square
            if self.board.board[ax][ay].unit != None and self.board.board[ax][ay].unit.color == self.active_player_color \
                    and self.board.board[ax][ay].unit.type.spawn:
                adjacent_spawner = True
        if not adjacent_spawner: return False, "Not adjacent to spawner"
        if not modify:
            return True, None
        # purchase unit
        self.money[self.active_player_color] -= cost
        # add unit to board
        self.board.board[x][y].add_unit(Unit(self.active_player_color, unit_type))
        return True, None

    def end_spawn_phase(self):
        assert self.phase == Phase.SPAWN, f"Tried to end spawn phase during phase {self.phase}"
        self.phase = Phase.TURN_END
        # collect money
        income = self.income_bonus
        for _, hex in self.board.hexes():
            if hex.unit != None and hex.unit.color == self.active_player_color and hex.is_graveyard:
                income += 1
        self.money[self.active_player_color] += income

        for unit, _ in self.units_with_locations():
            unit.reset_spawn_phase_end()

    def end_move_phase(self):
        assert self.phase == Phase.MOVE, f"Tried to end move phase during phase {self.phase}"
        self.phase = Phase.SPAWN

        for unit, _ in self.units_with_locations():
            unit.reset_move_phase_end()

    def full_turn(self, action_list: ActionList, verbose=False):
        """
        Process a full turn of the game from a list of actions
        Note that after this is called, the other player is active.
        """
        assert self.phase == Phase.MOVE, f"Tried to full turn during phase {self.phase}"
        if verbose:
            print(f"Processing actionlist: {action_list}")
        for action in action_list.move_phase:
            success, error_msg = self.process_single_action(action)
            if verbose and not success:
                print(f"Failed to process action: {action} -- {error_msg}")
        self.end_move_phase()
        for action in action_list.spawn_phase:
            success, error_msg = self.process_single_spawn(action)
            if verbose and not success:
                print(f"Failed to process action: {action} -- {error_msg}")
        self.end_spawn_phase()

    def copy(self):
        """
        Returns a copy of the game, for handing to an Agent to analyze.
        Temporary Unit properties (damage, etc) are NOT copied; only use this at turn start.
        """
        return Game(money=self.money.copy(),
                    max_turns=self.remaining_turns, 
                    board=self.board.copy(), 
                    active_player_color=self.active_player_color, 
                    phase=self.phase,
                    income_bonus=self.income_bonus,
                    new_game=False,
                    record_metrics=False)

    def encode_json(self):
        """
        Returns a JSON-encodable representation of the game.
        """
        return {
            "board": self.board.encode_json(),
            "money": self.money,
            "max_turns": self.remaining_turns,
            "active_player_color": self.active_player_color,
            "phase": self.phase,
            "income_bonus": self.income_bonus,
        }

    @staticmethod
    def decode_json(json):
        """
        Returns a Game object from a JSON-encoded representation.
        """
        return Game(money=json["money"],
                    max_turns=json["max_turns"],
                    board=Board.decode_json(json["board"]),
                    active_player_color=json["active_player_color"],
                    phase=Phase(json["phase"]),
                    income_bonus=json["income_bonus"],
                    new_game=False)

class Unit():
    def __init__(self, color, unit_type):
        self.type = unit_type
        self.color = color
        self.curr_health = self.type.defense
        self.hasMoved = False
        self.remainingAttack = self.type.attack
        self.isExhausted = False
        self.is_soulbound = False
    
    def reset_move_phase_end(self):
        self.hasMoved = False
        self.remainingAttack = self.type.attack
        self.curr_health = self.type.defense

    def reset_spawn_phase_end(self):
        pass

    def receive_attack(self, attack):
        self.curr_health -= attack
        if self.curr_health <= 0:
            if self.is_soulbound:
                return -2
            return self.type.rebate
        return -1

    def encode_json(self):
        return {
            "type": self.type.name,
            "color": self.color,
            "curr_health": self.curr_health,
            "hasMoved": self.hasMoved,
            "remainingAttack": self.remainingAttack,
            "isExhausted": self.isExhausted,
            "is_soulbound": self.is_soulbound,
        }
    
    @staticmethod
    def decode_json(json):
        type = unit_type_from_name(json["type"])
        unit = Unit(json["color"], type)
        unit.curr_health = json["curr_health"]
        unit.hasMoved = json["hasMoved"]
        unit.remainingAttack = json["remainingAttack"]
        unit.isExhausted = json["isExhausted"]
        unit.is_soulbound = json["is_soulbound"]
        return unit

    def copy(self):
        u = Unit(self.color, self.type)
        u.curr_health = self.curr_health
        u.hasMoved = self.hasMoved
        u.remainingAttack = self.remainingAttack
        u.isExhausted = self.isExhausted
        u.is_soulbound = self.is_soulbound
        return u

    def __hash__(self) -> int:
        return hash((self.color, self.type.name, self.curr_health, self.hasMoved, self.remainingAttack, self.isExhausted, self.is_soulbound))

def print_n_games(games):
    width = 15
    height = 9
    result = [""] * height
    for game in games:
        game_pretty_print = game.pretty_print(do_print=False).split("\n")
        assert len(game_pretty_print) == height, f"Expected {height} lines, got {len(game_pretty_print)}:\n{game.pretty_print}"
        for i in range(height):
            result[i] += game_pretty_print[i][:width].ljust(width) + "|"
    print("\n".join(result))
