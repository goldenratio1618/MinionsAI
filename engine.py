import copy
import enum
import random
import sys
import subprocess
import threading

BOARD_SIZE = 3
INCOME_BONUS = 3
MAX_TURNS = 100

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
            unit_index = self.board[i][j].unit.index
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
    MOVE = "move"
    SPAWN = "spawn"
    TURN_END = "turn_end"

class Game():
    def __init__(self, p0_money=0, p1_money=0, max_turns=MAX_TURNS):
        # starting position: captains on opposite corners with one graveyard in center
        self.graveyard_locs = [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)]
        self.water_locs = []
        self.board = Board(self.water_locs, self.graveyard_locs)
        self.board.board[0][0].add_unit(Unit(0, 0)) # yellow captain
        self.board.board[BOARD_SIZE - 1][ BOARD_SIZE - 1].add_unit(Unit(1, 0)) # blue captain
        #self.board.board[1][1].add_unit(Unit(1, 1)) # blue zombie
        # money for two sides
        self.money = [p0_money, p1_money]
        self.active_player_color = 0
        self.phase = Phase.TURN_END

        self.backup_for_undo = None
        self.remaining_turns = max_turns

        self.next_turn()

    @property
    def done(self):
        return self.remaining_turns <= 0

    @property
    def winner(self):
        assert self.done
        return 0 if self.money[0] > self.money[1] else 1

    @property
    def inactive_player_color(self):
        return 1 - self.active_player_color

    def units_with_locations(self, color=None):
        result = []
        for (i, j), hex in self.board.hexes():
            if hex.unit is not None and (color is None or hex.unit.color == color):
                result.append([hex.unit, (i, j)])
        return result

    def next_turn(self):
        self.active_player_color = self.inactive_player_color
        self.remaining_turns -= 1
        if self.done:
            return

        for row in self.board.board:
            for square in row:
                if square.unit != None and square.unit.color == self.active_player_color:
                    square.unit.hasMoved = False
                    square.unit.remainingAttack = unitList[square.unit.index].attack
                    square.unit.curr_health = unitList[square.unit.index].defense
        self.backup_for_undo = {
            'board': copy.deepcopy(self.board),
            'money': copy.deepcopy(self.money),
        }

    def process_single_move(self, move_action) -> bool:
        # returns true if move is legal & succesful, false otherwise

        assert self.phase == Phase.MOVE, f"Tried to move during phase {self.phase}"
        # TODO: Make sure there is a path from origin to destination
        xi, yi, xf, yf = move_action
        # make sure from hex has unit that can move
        if self.board.board[xi][yi].unit == None: return False
        # only move your own units
        if self.board.board[xi][yi].unit.color != self.active_player_color: return False
        # make sure origin and destination are sufficiently close
        speed = unitList[self.board.board[xi][yi].unit.index].speed
        attack_range = unitList[self.board.board[xi][yi].unit.index].attack_range
        distance = dist(xi, yi, xf, yf)
        # if target hex is empty parse move as movement
        if self.board.board[xf][yf].unit == None:
            if distance > speed: return False
            if self.board.board[xi][yi].unit.hasMoved: return False
            if not unitList[self.board.board[xi][yi].unit.index].flying and self.board.board[xf][yf].is_water: return False
            self.board.board[xi][yi].unit.hasMoved = True
            if unitList[self.board.board[xi][yi].unit.index].lumbering:
                self.board.board[xi][yi].unit.remainingAttack = 0
            self.board.board[xf][yf].add_unit(self.board.board[xi][yi].unit)
            self.board.board[xi][yi].remove_unit()
        # if target hex is occupied by friendly unit then swap the units
        elif self.board.board[xf][yf].unit.color == self.active_player_color:
            if distance > speed: return False
            if distance > unitList[self.board.board[xf][yf].unit.index].speed: return False
            if self.board.board[xi][yi].unit.hasMoved: return False
            if self.board.board[xf][yf].unit.hasMoved: return False
            if not unitList[self.board.board[xi][yi].unit.index].flying and self.board.board[xf][yf].is_water: return False
            if not unitList[self.board.board[xf][yf].unit.index].flying and self.board.board[xi][yi].is_water: return False
            self.board.board[xi][yi].unit.hasMoved = True
            self.board.board[xf][yf].unit.hasMoved = True
            if unitList[self.board.board[xi][yi].unit.index].lumbering:
                self.board.board[xi][yi].unit.remainingAttack = 0
            if unitList[self.board.board[xf][yf].unit.index].lumbering:
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
            if unitList[self.board.board[xi][yi].unit.index].unsummoner and not unitList[self.board.board[xf][yf].unit.index].persistent:
                self.money[self.inactive_player_color] += unitList[self.board.board[xf][yf].unit.index].cost
                self.board.board[xf][yf].remove_unit()
            # attacking prevents later movement
            self.board.board[xi][yi].unit.hasMoved = True
            # flurry deals 1 attack
            if unitList[self.board.board[xi][yi].unit.index].flurry:
                self.board.board[xi][yi].unit.remainingAttack -= 1
                attack_outcome = self.board.board[xf][yf].unit.receive_attack(1)
            # otherwise deal full attack
            else:
                self.board.board[xi][yi].unit.remainingAttack = 0
                attack_outcome = self.board.board[xf][yf].unit.receive_attack(unitList[self.board.board[xi][yi].unit.index].attack)
            # process dead unit, if applicable
            if attack_outcome >= 0:
                # remove unit from board
                self.board.board[xf][yf].remove_unit()
                # process rebate
                self.money[self.inactive_player_color] += attack_outcome
            return True

    def process_single_spawn(self, spawn_action) -> bool:
        # returns true if spawn is legal & succesful, false otherwise

        assert self.phase == Phase.SPAWN, f"Tried to move during phase {self.phase}"
        # TODO: treat reinforcements explicitly (so that one can spawn bounced units without paying dollars)
        index, x, y = spawn_action
        # check to see if we have enough money
        cost = unitList[index].cost
        if cost > self.money[self.active_player_color]: return False
        # check to make sure hex is unoccupied
        if self.board.board[x][y].unit != None: return False
        # check to make sure we are adjacent to spawner
        adjacent_spawner = False
        for square in adjacent_hexes(x, y):
            ax, ay = square
            if self.board.board[ax][ay].unit != None and self.board.board[ax][ay].unit.color == self.active_player_color \
                    and unitList[self.board.board[ax][ay].unit.index].spawn:
                adjacent_spawner = True
        if not adjacent_spawner: return False
        # purchase unit
        self.money[self.active_player_color] -= cost
        # add unit to board
        self.board.board[x][y].add_unit(Unit(self.active_player_color, index))
        return True

    def turn(self, move_list, spawn_list, auto_continue=True):
        assert not self.done

        # parse all moves
        self.phase = Phase.MOVE
        for move in move_list:
            self.process_single_move(move)
        
        # parse all spawns
        self.phase = Phase.SPAWN
        for spawn_action in spawn_list:
            self.process_single_spawn(spawn_action)

        self.phase = Phase.TURN_END
        # collect money
        income = INCOME_BONUS
        for square in self.graveyard_locs:
            x, y = square
            if self.board.board[x][y].unit != None and self.board.board[x][y].unit.color == self.active_player_color:
                income += 1
        self.money[self.active_player_color] += income

        if auto_continue:
            self.next_turn()

    def undo(self):
        self.board = self.backup_for_undo['board']
        self.money = self.backup_for_undo['money']
        self.backup_for_undo = None

class UnitType():
    def __init__(self, attack, defense, speed, attack_range, persistent, immune, max_stack, spawn, blink, unsummoner, deadly, flurry, flying, lumbering, terrain_ability, cost, rebate):
        self.attack = attack
        self.defense = defense
        self.speed = speed
        self.attack_range = attack_range
        self.persistent = persistent
        self.immune = immune
        self.max_stack = max_stack
        self.spawn = spawn
        self.blink = blink
        self.unsummoner = unsummoner
        self.deadly = deadly
        self.flurry = flurry
        self.flying = flying
        self.lumbering = lumbering
        self.terrain_ability = terrain_ability
        self.cost = cost
        self.rebate = rebate

unitList = [
    UnitType(0, 7, 1, 1, True, True, 1, True, False, True, False, False, False, False, 0, 255, 0), # captain
    UnitType(1, 2, 1, 1, False, False, 1, False, False, False, False, False, False, True, 0, 2, 0) # zombie
]

class Unit():
    def __init__(self, color, index):
        self.index = index
        self.color = color
        self.curr_health = unitList[index].defense
        self.hasMoved = True
        self.remainingAttack = 0
        self.isExhausted = True
        self.is_soulbound = False
    
    def receive_attack(self, attack):
        self.curr_health -= attack
        if self.curr_health <= 0:
            if self.is_soulbound:
                return -2
            return unitList[self.index].rebate
        return -1

# moves are in the form (xi, yi, xf, yf)
# move list is terminated by an empty line
def parse_input(proc):
    input_list = []
    line = proc.stdout.readline().strip()
    while line != "":
        input_list.append([int(s) for s in line.split(" ") if s != ""])
        line = proc.stdout.readline().strip()
    return input_list

# actual game
def main():
    game = Game(0, 6)
    game.board.print_board_properties()
    print()
    game.board.print_board_state()
    print()

    yellow = subprocess.Popen(["python3", "-u", "randomAI.py"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
    blue = subprocess.Popen(["python3", "-u", "randomAI.py"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)

    # turn loop
    while not game.done:
        # yellow turn -- get input from screen
        yellow.stdin.write("Your turn\n")
        yellow.stdin.flush()
        move_list = parse_input(yellow)
        spawn_list = parse_input(yellow)
        #xi = random.randrange(0, 5)
        #yi = random.randrange(0, 5)
        #xf = random.randrange(0, 5)
        #yf = random.randrange(0, 5)
        #move_list = [(xi, yi, xf, yf)]
        #spawn_list = []
        game.turn(move_list, spawn_list)
        game.board.print_board_state()
        print()

        # blue turn -- pass
        blue.stdin.write("Your turn\n")
        blue.stdin.flush()
        move_list = parse_input(blue)
        spawn_list = parse_input(blue)
        game.turn(move_list, spawn_list)
        game.board.print_board_state()
        print()
    # print final-state money
    print("Game over!")
    print(game.money[0], game.money[1])
    # write final board state to file
    with open("output.txt", "w") as f:
        original_stdout = sys.stdout
        sys.stdout = f
        print(game.money[0], game.money[1])
        print()
        game.board.print_board_state()
        sys.stdout = original_stdout
    # throw exception if yellow loses (only way to get output to screen)
    if (game.money[0] < game.money[1]):
        raise Exception("Yellow loses!  Yellow money = " + str(game.money[0])
            + " but blue money = " + str(game.money[1]))
    
if __name__ == "__main__":
  main()
