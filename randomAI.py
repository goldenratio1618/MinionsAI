# random AI
# move list is random hex from random other hex
# spawn list is zombie on random hex

import random

def main():
    while input() != "":
        continue

    while True:
        line = input()
        while line != "":
            if line == "Game over!":
                return
            line = input()
        xi = random.randrange(0, 5)
        yi = random.randrange(0, 5)
        xf = random.randrange(0, 5)
        yf = random.randrange(0, 5)
        print(xi, yi, xf, yf, flush=True)
        print("", flush=True)
        print("", flush=True)

if __name__ == "__main__":
  main()
