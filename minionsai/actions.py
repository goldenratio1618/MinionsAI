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
    if all(v == 0 for v in state[action[2],action[3],:] == 0):
        state[action2,action3]
        if state[action[2],action[3]] > 0:
            reward = -1
        else:
            reward = 1
            
            

print(len(possibly_legal_actions()))
