# Launching from scratch on new machine:
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
