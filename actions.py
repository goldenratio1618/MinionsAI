class Board():
    def __init__(board_size, graveyard_locs, water_locs):
        self.board = [[Hex((i,j) in water_locs, (i,j) in graveyard_locs) for j in range(board_size[1])] for i in range(board_size[0])]






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



class Unit():
    def __init__(self, color, attack, defense, speed, range, persistent, immune, max_stack, spawn, blink, unsummoner, deadly, flying, lumbering, terrain_ability, cost, rebate):
        self.color = color
        self.attack = attack
        self.defense = defense
        self.curr_health = defense
        self.speed = speed
        self.remainingSpeed = speed
        self.remainingAttack = 0
        self.isExhausted = 1
        self.range = range
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
        self.is_soulbound = False
    
    def receive_attack(self, unit):
        self.curr_health -= unit.attack
        if self.curr_health < 0:
            if self.is_soulbound:
                return -2
            return self.rebate
        return -1

class Zombie(Unit):
    def __init__(self, color):
        super().__init__(color, 1, 2, 1, 1, False, False, 1, False, False, False, False, False, True, 0, 2, 0)

class Necromancer(Unit):
    def __init__(self, color):
        super().__init__(color, 0, 2, 1, 1, True, True, 1, True, False, True, False, False, True, 0, 0, -128)


    





def possibly_legal_actions():
    possibly_legal = []
    for i in range(BOARD_SIZE[0]):
        for j in range(BOARD_SIZE[1]):
            for x in range(BOARD_SIZE[0]):
                if abs(x-i) > 3:
                    continue
                for y in range(BOARD_SIZE[1]):
                    if abs(y-j) > 3 and (j != 1 or y != 1):
                        continue
                    possibly_legal.append((i,j,x,y))
    return possibly_legal

def process_action(action, state):
    reward = 0
    # not implemented yet
    if 0 in [action[1], action[3]]:
        return reward
    if all(v == 0 for v in state[action[0],action[1],:]):
        return reward
    if all(v == 0 for v in state[action[2],action[3],:] == 0:
        state[action2,action3]
        if state[action[2],action[3]] > 0:
            reward = -1
        else:
            reward = 1
            
            

print(len(possibly_legal_actions()))