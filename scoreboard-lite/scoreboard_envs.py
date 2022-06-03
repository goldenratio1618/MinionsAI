from minionsai.engine import Game

ENVS = {}

def zombies5x5():
    return Game(money=(2, 4),
                max_turns=20,
                income_bonus=1,)
ENVS['zombies5x5'] = zombies5x5