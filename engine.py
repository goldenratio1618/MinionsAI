BOARD_SIZE = 5

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
    
    def remove_unit(self, unit):
        self.unit = None
    
class Game():
    def __init__(self, p1_money, p2_money):
        # starting position: captains on opposite corners with one graveyard in center
        graveyard_locs = [(2, 3)]
        water_locs = []
        self.board = Board(water_locs, graveyard_locs)
        self.board.board[0][0].add_unit(Unit(0, 0)) # yellow captain
        self.board.board[4][4].add_unit(Unit(1, 0)) # blue captain

class UnitType():
    def __init__(self, attack, defense, speed, attack_range, persistent, immune, max_stack, spawn, blink, unsummoner, deadly, flying, lumbering, terrain_ability, cost, rebate):
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
        self.flying = flying
        self.lumbering = lumbering
        self.terrain_ability = terrain_ability
        self.cost = cost
        self.rebate = rebate

unitList = [
    UnitType(0, 7, 1, 1, True, True, 1, True, False, True, False, False, False, 0, 255, 0), # captain
    UnitType(1, 2, 1, 1, False, False, 1, False, False, False, False, False, True, 0, 2, 0) # zombie
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
    
    def receive_attack(self, unit):
        self.curr_health -= unit.attack
        if self.curr_health < 0:
            if self.is_soulbound:
                return -2
            return self.rebate
        return -1

# actual game
game = Game(0, 6)
game.board.print_board_properties()
print()
game.board.print_board_state()
