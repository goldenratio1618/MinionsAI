import enum
import random
from typing import Tuple
from .unit_type import UnitType, NECROMANCER, ZOMBIE
from .action import ActionType, Action, ActionList, MoveAction, SpawnAction
import numpy as np

BOARD_SIZE = 5

# distance function between two hexes
def dist(xi, yi, xf, yf):
    return max(abs(yf - yi), abs(xf - xi) + (abs(yf - yi) if xi > xf == yi > yf else 0))

# return an array of tuples of adjacent hexes
def adjacent_hexes(x, y):
    hex_list = []
    if x > 0: hex_list.append((x-1,y))
    if x < (BOARD_SIZE - 1): hex_list.append((x+1,y))
    if y > 0:
        hex_list.append((x,y-1))
        if x < (BOARD_SIZE - 1): hex_list.append((x+1,y-1))
    if y < (BOARD_SIZE - 1):
        hex_list.append((x,y+1))
        if x > 0: hex_list.append((x-1,y+1))
    return hex_list

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
                b.board[i][j].add_unit(Unit(color=hex.unit.color, unit_type=hex.unit.type))
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

class Phase(enum.Enum):
    MOVE = "move"  # Move phase
    SPAWN = "spawn"  # Spawn Phase
    TURN_END = "turn_end"  # After spawn phase, but haven't yet run next_turn()
    GAME_OVER = "game_over"  # Game is done.

class Game():
    def __init__(self, 
                 money=(4, 8),
                 max_turns=20, 
                 board=None, 
                 active_player_color=0, 
                 phase=Phase.TURN_END,
                 income_bonus=1):
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
            graveyard_locs = [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if random.random() < 0.25 and 1 <= i + j and i + j <= (2 * BOARD_SIZE - 3)]
            water_locs = []
            board = Board(water_locs, graveyard_locs)
            board.board[0][0].add_unit(Unit(0, NECROMANCER)) # yellow captain
            board.board[BOARD_SIZE - 1][ BOARD_SIZE - 1].add_unit(Unit(1, NECROMANCER)) # blue captain
        self.board: Board = board

        self.income_bonus: int = income_bonus

        # money for two sides
        self.money = list(money)
        self.active_player_color: int = active_player_color
        self.phase: Phase = phase

        self.remaining_turns: int = max_turns

    @property
    def done(self) -> bool:
        return self.remaining_turns <= 0

    @property
    def winner(self) -> int:
        """
        Returns index of winning player (0 or 1)
        """
        assert self.done
        return 0 if self.money[0] > self.money[1] else 1

    @property
    def inactive_player_color(self) -> int:
        return 1 - self.active_player_color

    def pretty_print(self):
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

        print("\n".join(row_strs))


    def units_with_locations(self, color=None) -> Tuple[Tuple[int, int], Hex]:
        result = []
        for (i, j), hex in self.board.hexes():
            if hex.unit is not None and (color is None or hex.unit.color == color):
                result.append([hex.unit, (i, j)])
        return result

    def next_turn(self) -> None:
        assert self.phase == Phase.TURN_END, f"Can only call game.next_turn() from TURN_END phase; received call in {self.phase.name}. Did you forget to call game.end_spawn_phase()?"
        self.active_player_color = self.inactive_player_color
        self.remaining_turns -= 1
        if self.done:
            self.phase = Phase.GAME_OVER
            return

        for row in self.board.board:
            for square in row:
                if square.unit != None and square.unit.color == self.active_player_color:
                    square.unit.hasMoved = False
                    square.unit.remainingAttack = square.unit.type.attack
                    square.unit.curr_health = square.unit.type.defense
        self.phase = Phase.MOVE

    def process_single_action(self, action: Action) -> bool:
        if self.phase == Phase.MOVE:
            # Note: We need to compare the names not the enum types,
            # in case the agent submitted an action from their local copy of teh codebase
            # which is technically a different enum type.
            if action.action_type.name == ActionType.MOVE.name:
                return self.process_single_move(action)
            else:
                raise ValueError(f"Wrong action type ({action.action_type}) for Move Phase. Did you forget to call game.end_spawn_phase() before moving on to the spawn_phase?")
        elif self.phase == Phase.SPAWN:
            if action.action_type.name == ActionType.SPAWN.name:
                return self.process_single_spawn(action)
            else:
                raise ValueError(f"Wrong action type ({action.action_type}) for Spawn Phase.")
        elif self.phase == Phase.TURN_END:
            raise ValueError("Tried to process actions during TURN_END phase. Did you forget to call game.next_turn() before the next player's actions?")
        else:
            raise ValueError(f"Wrong phase ({self.phase}) for processing actions.")

    def process_single_move(self, move_action: MoveAction) -> bool:
        # returns true if move is legal & succesful, false otherwise

        assert self.phase == Phase.MOVE, f"Tried to move during phase {self.phase}"
        # TODO: Make sure there is a path from origin to destination
        xi, yi = move_action.from_xy
        xf, yf = move_action.to_xy
        # make sure from hex has unit that can move
        if self.board.board[xi][yi].unit == None: return False
        # only move your own units
        if self.board.board[xi][yi].unit.color != self.active_player_color: return False
        # make sure origin and destination are sufficiently close
        speed = self.board.board[xi][yi].unit.type.speed
        attack_range = self.board.board[xi][yi].unit.type.attack_range
        distance = dist(xi, yi, xf, yf)
        # if target hex is empty parse move as movement
        if self.board.board[xf][yf].unit == None:
            if distance > speed: return False
            if self.board.board[xi][yi].unit.hasMoved: return False
            if not self.board.board[xi][yi].unit.type.flying and self.board.board[xf][yf].is_water: return False
            self.board.board[xi][yi].unit.hasMoved = True
            if self.board.board[xi][yi].unit.type.lumbering:
                self.board.board[xi][yi].unit.remainingAttack = 0
            self.board.board[xf][yf].add_unit(self.board.board[xi][yi].unit)
            self.board.board[xi][yi].remove_unit()
        # if target hex is occupied by friendly unit then swap the units
        elif self.board.board[xf][yf].unit.color == self.active_player_color:
            if distance > speed: return False
            if distance > self.board.board[xf][yf].unit.type.speed: return False
            if self.board.board[xi][yi].unit.hasMoved: return False
            if self.board.board[xf][yf].unit.hasMoved: return False
            if not self.board.board[xi][yi].unit.type.flying and self.board.board[xf][yf].is_water: return False
            if not self.board.board[xf][yf].unit.type.flying and self.board.board[xi][yi].is_water: return False
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
            if distance > attack_range: return False
            if self.board.board[xi][yi].unit.remainingAttack == 0: return False
            # unsummon removes non-persistent unit from board and refunds cost
            if self.board.board[xi][yi].unit.type.unsummoner and not self.board.board[xf][yf].unit.type.persistent:
                self.money[self.inactive_player_color] += self.board.board[xf][yf].unit.type.cost
                self.board.board[xf][yf].remove_unit()
            # attacking prevents later movement
            self.board.board[xi][yi].unit.hasMoved = True
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
        return True

    def process_single_spawn(self, spawn_action: SpawnAction) -> bool:
        # returns true if spawn is legal & succesful, false otherwise

        assert self.phase == Phase.SPAWN, f"Tried to spawn during phase {self.phase}"
        # TODO: treat reinforcements explicitly (so that one can spawn bounced units without paying dollars)
        unit_type: UnitType = spawn_action.unit_type
        x, y = spawn_action.to_xy
        # check to see if we have enough money
        cost = unit_type.cost
        if cost > self.money[self.active_player_color]: return False
        # check to make sure hex is unoccupied
        if self.board.board[x][y].unit != None: return False
        # check to make sure we are adjacent to spawner
        adjacent_spawner = False
        for square in adjacent_hexes(x, y):
            ax, ay = square
            if self.board.board[ax][ay].unit != None and self.board.board[ax][ay].unit.color == self.active_player_color \
                    and self.board.board[ax][ay].unit.type.spawn:
                adjacent_spawner = True
        if not adjacent_spawner: return False
        # purchase unit
        self.money[self.active_player_color] -= cost
        # add unit to board
        self.board.board[x][y].add_unit(Unit(self.active_player_color, unit_type))
        return True

    def end_spawn_phase(self):
        assert self.phase == Phase.SPAWN, f"Tried to end spawn phase during phase {self.phase}"
        self.phase = Phase.TURN_END
        # collect money
        income = self.income_bonus
        for _, hex in self.board.hexes():
            if hex.unit != None and hex.unit.color == self.active_player_color and hex.is_graveyard:
                income += 1
        self.money[self.active_player_color] += income

    def end_move_phase(self):
        assert self.phase == Phase.MOVE, f"Tried to end move phase during phase {self.phase}"
        self.phase = Phase.SPAWN

    def full_turn(self, action_list: ActionList, verbose=False):
        """
        Process a full turn of the game from a list of actions
        Note that after this is called, the other player is active.
        """
        assert self.phase == Phase.MOVE, f"Tried to full turn during phase {self.phase}"
        if verbose:
            print(f"Processing actionlist: {action_list}")
        for action in action_list.move_phase:
            success = self.process_single_action(action)
            if verbose and not success:
                print(f"Failed to process action: {action}")
        self.end_move_phase()
        for action in action_list.spawn_phase:
            success = self.process_single_spawn(action)
            if verbose and not success:
                print(f"Failed to process action: {action}")
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
                    income_bonus=self.income_bonus)

class Unit():
    def __init__(self, color, unit_type):
        self.type = unit_type
        self.color = color
        self.curr_health = self.type.defense
        self.hasMoved = False
        self.remainingAttack = 0
        self.isExhausted = False
        self.is_soulbound = False
    
    def receive_attack(self, attack):
        self.curr_health -= attack
        if self.curr_health <= 0:
            if self.is_soulbound:
                return -2
            return self.type.rebate
        return -1