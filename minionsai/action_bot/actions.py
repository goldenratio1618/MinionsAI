from attr import NOTHING
from minionsai.action import MoveAction, SpawnAction
from minionsai.engine import BOARD_SIZE
from minionsai.unit_type import ZOMBIE


def possibly_legal_moves():
    possibly_legal = []
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if abs(x-i) > 1:
                    continue
                for y in range(BOARD_SIZE):
                    if abs(y-j) > 1 and (j != 1 or y != 1):
                        continue
                    possibly_legal.append(MoveAction((i,j),(x,y)))
    
    return possibly_legal

def possibly_legal_spawns():
    possibly_legal = []
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            possibly_legal.append(SpawnAction(ZOMBIE, (i,j)))
    return possibly_legal

def process_action(action, state):
    reward = 0
    # not implemented yet
    if 0 in [action[1], action[3]]:
        return reward
    if all(v == 0 for v in state[action[0],action[1],:]):
        return reward
    if all(v == 0 for v in state[action[2],action[3],:] == 0):
        state[action2,action3]
        if state[action[2],action[3]] > 0:
            reward = -1
        else:
            reward = 1
            
            

print(len(possibly_legal_actions()))
