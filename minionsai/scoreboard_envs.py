from minionsai.engine import Game

ENVS = {}

def zombies5x5():
    return Game(money=(3, 4),
                max_turns=20,
                income_bonus=1,
                symmetrize=True,
                min_graveyards=3,
                max_graveyards=8)
ENVS['zombies5x5'] = zombies5x5
