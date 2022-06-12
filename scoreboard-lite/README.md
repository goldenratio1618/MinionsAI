# Overview
The scoreboard is really two separate entities that live on the same machine:
1. The *webserver* is a little flask app that serves pages to users. Flask is quite simple (I didn't know any flask before I started writing this). It's all defined in `app.py`.
2. The *ts_worker* runs games among the agents forever and updates their trueskill values.
Eventually we'll want one ts_worker per environment, I suspect.

The two communicate by (a) the webserver loading agents into the `active_agents` directory which the ts_worker watches and (2) the ts_worker writes its state to the `scores.csv` file which the webserver reads.

# Launching from scratch on new machine
```
mkdir code
cd code

git clone https://github.com/goldenratio1618/MinionsAI.git
cd MinionsAI

pip install -e .

cd scoreboard-lite
pip3 install -r extra_requirements.txt
pip3 install torch --no-cache-dir
mkdir data
tmux
```

# tmux commands
```
>tmux           -- make a new tmux
>tmux a         -- connect to already-running tmux
ctrl-b d        -- disconnect from the tmux
ctrl-b <arrow>  -- move around the panes
```
https://tmuxcheatsheet.com/

# Run threads
Run each of these from their own tmux

Launch webserver
```
cd scoreboard-lite
python3 -m flask run --host=0.0.0.0 --port=80
```

Launch env runner
```
cd scoreboard-lite
python3 scoreboard_ts_worker.py ENV_NAME
```

# Security
Hahahaha

# Envs
Envs are defined in `scoreboard_envs.ENVS`. We should add new envs rarely, so for now it's a pretty manual process. To do it, you need to add a new one there, then ssh to the server and restart the runner.
