# random AI
# move list is random hex from random other hex
# spawn list is zombie on random hex

import random

BOARD_SIZE=5

def main():
    while input() != "":
        continue

    while True:
        line = input()
        while line != "":
            if line == "Game over!":
                return
            line = input()
        # random movement
        xi = random.randrange(0, BOARD_SIZE)
        yi = random.randrange(0, BOARD_SIZE)
        xf = random.randrange(0, BOARD_SIZE)
        yf = random.randrange(0, BOARD_SIZE)
        print(xi, yi, xf, yf, flush=True)
        print("", flush=True)
        
        # random spawning
        x = random.randrange(0, BOARD_SIZE)
        y = random.randrange(0, BOARD_SIZE)
        print(1, x, y, flush=True)
        print("", flush=True)

if __name__ == "__main__":
  main()
