# MinionsAI

## Installation & setup
Install python stuff with via pip:
```
>cd MinionsAI
>pip install -e .
```

Check that it worked by running a game:
```
python scripts/run_game.py
```

Check that you can run the tests:
```
> python -m pytest tests/
```

If you want to run ML experiments, set up an experiments directory by making a file at `minionsai/local_config.py`, containing:
```
EXPERIMENTS_DIRECTORY = "/path/to/directory"
```
If you don't do this it will use your system's temp directory, which is also fine.

## Available scripts
* `run_game.py`: Play a game between two random agents. Mostly just for testing.
* `play_vs_agent.py`: Let's you play on the command line against a random agent (or any other agent if you have one).
* `ts_worker.py`: Evaluates all agents in a directory against one another
* `eval_two_agents.py`: Runs two agents against each other once or many times.

## On Imports
Due to our agent serialization, there are a couple rules on imports:

1. Within any library code that will be run in an agent, only use *relative* imports, such as:
```
from ..engine import game
```
If you try to do `from minionsai.engine` instead, when your agent is serialized it may look for the system level package rather than the serialized one.

2. This means that all our scripts (things you'd run as `__main__`, have to live *outside* the `minionsai` folder, in `scripts/` or `tests/`). 

If you try to run something inside `minionsai` as `__main__`, you'll get:
```
ImportError: attempted relative import with no known parent package
```

I found the first answer [here](https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time) helpful for explaining this stuff

## Writing an Agent
To write an agent you'll need a subclass of `Agent`.
You need to implement the `act()` function for your subclass,
which should take a `Game` object representing the start of the turn and return an `ActionList` of the actions you want to take on your turn.

Inside the `act()` function you can do whatever you want - regular python logic, call out to C++, feed forward a neural net, etc etc.

## Using the Game object (as an Agent)
The game object contains various properties like `game.money` and `game.board` that describe the state of the game. 

You can also use `game.process_single_action`, `game.full_turn()`, etc, to see the state the game would have if you tried various actions.

Use `game.copy()` at the start if you want a backup copy, to try out several possible turns.

Nothing you do to your `game` object will affect the actual game you're playing (your object is a copy). All the matters is waht you return in your final `ActionList`

## Using the Game object (as control code)
You may want to run your own games (e.g. for training).
Check out `run_game.py` for an example which explains the way to do it.

* Before each turn, call `game.next_turn()`. This causes the game to switch the active player, refresh all units, check for victory, etc.
* After `game.next_turn()`, you probably want to check `game.done` and do something special if it's True.
* After that you need to proces the active player's turn. There are two ways to do that: processing an entire turn of actions all at once, or one by one.

Examples:
```
# Run a game processing turns all at once
game = Game()
while True:
    game.next_turn()
    if game.done:
        break
    action_list = ...
    game.full_turn(action_list)


# Run a game doing all the steps manually, one action at a time:
game = Game()
while True:
    game.next_turn()
    if game.done:
        break
    
    # State Move Phase
    game.proces_single_action(action1)
    game.proces_single_action(action2)
    game.proces_single_action(action3)

    game.end_move_phase()

    game.process_single_action(action3)
    game.process_single_action(action4)

    game.end_spawn_phase()
```


# Agent Serialization
In order to play agents from different codebases against each other, we need to be able to save and load them.

Example lifetime:

```
agent = ExampleAgent()  # User subclass
train(agent)  # User training process
agent_saveload.save(directory)  # Save the agent.

# Now 3 days go by, codebase changes, but we want to compare this old agent to new one.
agent = agent_saveload.loaddirectory)  # Load the agent. Note that this will be an ExampleAgent, even though that class may no longer exist in the codebase.
run_game(game, agent, newer_agent)
```

We achieve this by saving a directory like this:

```
save_dir/
    code/
        <entire snapshot of the codebase>
    agent/
        <data necessary to reconstruct the agent instance>
    __init__.py    <-- contains build_agent() which rebuilds the agent.
```

## Submitting to scoreboard
Once you have a good agent, you can submit it to the scoreboard at `minions-scoreboard.com`

[Note: the scoreboard is incredibly janky, and if it's down poke David to fix it.]

To submit an agent, run `agent_saveload.save(directory)`. Then rename the directory to `<agent_name>`, zip it into `<agent_name>.zip` and upload it to the scoreboard. 
Make sure to give it a very unique name - recommend putting your name and the date in there at the very least.
Note that (for me at least), the agent name winds up being the name fo teh directory pre-zipping, so make sure to rename first.

## Restarting the Scoreboard
If the scoreboard is down, here is the minimal set of instructions to restart it:
1. ssh and enter the tmux
```
local> ssh root@minions-scoreboard.com
minions> tmux a 
```
2. Now you're in the tmux (which is a persistent terminal that survives your ssh tunnel ending). 
It's got two panes, which you can navigate between with `ctrl-b <uparrow>` and ```ctrl-b <downarrow>`. 
One pane is the web server and the other is the trueskill evaluator.
Both should be produing logs.
If either is not, you should be able to use `<uparrow>` in the appropriate pane to rerun the latest command.

## Profiling
```
python -m cProfile -o train.prof scripts/disc_only_train.py
python -m tuna train.prof
```
(After `pip install tuna`)