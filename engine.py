BOARD_SIZE = 5

# distance function between two hexes
# when y is decreased, x can stay the same or be increased by 1
def dist(xi, yi, xf, yf):
    if (yi > yf): return dist(xf, yf, xi, yi)
    if (yi == yf): return abs(xi - xf)
    if (xi > xf): return dist(xi, yi, xf + 1, yf - 1) + 1
    return (yf - yi) + (xf - xi)

class Board():
    def __init__(self, water_locs, graveyard_locs):
        self.board = [[Hex((i,j) in water_locs, (i,j) in graveyard_locs) for j in range(BOARD_SIZE)] for i in range(BOARD_SIZE)]

    def print_board_properties(self):
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                print(i, j, self.board[i][j].is_water, self.board[i][j].is_graveyard)

    def print_board_state(self):
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if (self.board[i][j].unit == None):
                    print(i, j)
                else:
                    print(i, j, self.board[i][j].unit.index, self.board[i][j].unit.color)
                
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
    
class Game():
    def __init__(self, p0_money, p1_money):
        # starting position: captains on opposite corners with one graveyard in center
        self.graveyard_locs = [(2, 3)]
        self.water_locs = []
        self.board = Board(self.water_locs, self.graveyard_locs)
        self.board.board[0][0].add_unit(Unit(0, 0)) # yellow captain
        self.board.board[4][4].add_unit(Unit(1, 0)) # blue captain
        # money for two sides
        self.money = [p0_money, p1_money]

    def turn(self, color, move_list, spawn_list):
        # reset move, attack, and health
        for row in self.board.board:
            for square in row:
                if square.unit != None and square.unit.color == color:
                    square.unit.hasMoved = False
                    square.unit.remainingAttack = unitList[square.unit.index].attack
                    square.unit.health = unitList[square.unit.index].defense
        # parse all moves
        # TODO: Make sure there is a path from origin to destination
        for move in move_list:
            xi, yi, xf, yf = move
            # make sure from hex has unit that can move
            if self.board.board[xi][yi].unit == None: continue
            # make sure origin and destination are sufficiently close
            speed = unitList[self.board.board[xi][yi].unit.index].speed
            attack_range = unitList[self.board.board[xi][yi].unit.index].attack_range
            distance = dist(xi, yi, xf, yf)
            # if target hex is empty parse move as movement
            if self.board.board[xf][yf].unit == None:
                if distance > speed: continue
                if self.board.board[xi][yi].unit.hasMoved: continue
                if not unitList[self.board.board[xi][yi].unit.index].flying and self.board.board[xf][yf].is_water: continue
                self.board.board[xi][yi].unit.hasMoved = True
                self.board.board[xf][yf].add_unit(self.board.board[xi][yi].unit)
                self.board.board[xi][yi].remove_unit()
            # if target hex is occupied by friendly unit then swap the units
            elif self.board.board[xf][yf].unit.color == color:
                if distance > speed: continue
                if distance > unitList[self.board.board[xf][yf].unit.index].speed: continue
                if self.board.board[xi][yi].unit.hasMoved: continue
                if self.board.board[xf][yf].unit.hasMoved: continue
                if not unitList[self.board.board[xi][yi].unit.index].flying and self.board.board[xf][yf].is_water: continue
                if not unitList[self.board.board[xf][yf].unit.index].flying and self.board.board[xi][yi].is_water: continue
                self.board.board[xi][yi].unit.hasMoved = True
                self.board.board[xf][yf].unit.hasMoved = True
                temp = self.board.board[xi][yi].unit
                self.board.board[xi][yi].remove_unit()
                self.board.board[xi][yi].add_unit(self.board.board[xf][yf].unit)
                self.board.board[xf][yf].remove_unit()
                self.board.board[xf][yf].add_unit(temp)
            # if target hex is occupied by enemy unit, then attack
            elif self.board.board[xf][yf].unit.color != color:
                if distance > attack_range: continue
                if self.board.board[xi][yi].unit.remainingAttack == 0: continue
                # attacking prevents later movement
                self.board.board[xi][yi].unit.hasMoved = True
                # flurry deals 1 attack
                if unitList[self.board.board[xi][yi].unit.index].flurry:
                    self.board.board[xi][yi].unit.remainingAttack -= 1
                    attack_outcome = self.board.board[xf][yf].unit.receiveAttack(1)
                # otherwise deal full attack
                else:
                    self.board.board[xi][yi].unit.remainingAttack = 0
                    attack_outcome = self.board.board[xf][yf].unit.receiveAttack(unitList[self.board.board[xi][yi].unit.index].attack)
                # process dead unit, if applicable
                if attack_outcome >= 0:
                    # remove unit from board
                    self.board.board[xf][yf].remove_unit()
                    # process rebate
                    self_money[1 - color] += attack_outcome


                


        

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
        if self.curr_health < 0:
            if self.is_soulbound:
                return -2
            return self.rebate
        return -1

# actual game
game = Game(0, 6)
#game.board.print_board_properties()
#print()
#game.board.print_board_state()

# turn 1: yellow moves captain from (0,0) to (0,1)
game.turn(0, [(0,0,0,1)], [])
game.board.print_board_state()
