class Board():
    def __init__(board_size, unit1, unit2):


class Unit():
    def __init__(self, color, attack, defense, speed, range, persistent, max_stack, spawn, blink, unsummoner, deadly, flying, lumbering, terrain_ability, cost, rebate, hex):
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
        self.hex = hex
        self.is_soulbound = False

    def move_unit(self, new_hex):
        self.hex = new_hex
        self.remainingSpeed = 0
    
    def receive_attack(self, unit):
        self.curr_health -= unit.attack
        if self.curr_health < 0:
            if self.is_soulbound:
                return -2
            return self.rebate
        return -1

    





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