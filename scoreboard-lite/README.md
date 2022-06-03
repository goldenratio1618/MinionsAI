TODO
* connect ports to external world
* better logging
  
# Launching from scratch on new machine:
```
mkdir code
cd code

git clone https://github.com/goldenratio1618/MinionsAI.git
cd MinionsAI

pip install -e .

cd scoreboard-lite
pip install -r extra_requirements.txt
mkdir data
tmux
```

# tmux commands
```
>tmux     -- make a new tmux
>tmux a   -- connect to already-running tmux
ctrl-b d  -- disconnect from the tmux
```

# Run threads
Run each of these from their own tmux

Launch webserver
```
cd scoreboard-lite
python3 -m flask run --host=0.0.0.0
```

Launch env runner
```
cd scoreboard-lite
python3 scoreboard_ts_worker.py ENV_NAME
```

