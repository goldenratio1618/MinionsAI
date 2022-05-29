# random AI
# move list is random hex from random other hex
# spawn list is zombie on random hex

import random
MAX_TURNS = 10000

while input() != "":
    continue

for i in range(MAX_TURNS):
    while input() != "":
        continue
    xi = random.randrange(0, 5)
    yi = random.randrange(0, 5)
    xf = random.randrange(0, 5)
    yf = random.randrange(0, 5)
    print(xi, yi, xf, yf, flush=True)
    print("", flush=True)
    print("", flush=True)
