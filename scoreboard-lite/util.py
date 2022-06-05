import os
from datetime import timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def list_envs():
    envs = os.listdir(DATA_DIR)
    envs = [e for e in envs if os.path.isdir(os.path.join(DATA_DIR, e))]
    return envs

def list_agents(env):
    agents = os.listdir(env_agents_dir(env))
    agents = [a for a in agents if os.path.isdir(os.path.join(env_agents_dir(env), a))]
    return agents

def env_dir(env_name):
    return os.path.join(DATA_DIR, env_name)

def env_agents_dir(env_name):
    return os.path.join(env_dir(env_name), 'active_agents')

def env_deleted_agents_dir(env_name):
    return os.path.join(env_dir(env_name), 'deleted_agents')

def format_timedelta(td: timedelta):
    seconds = int(td.total_seconds())
    periods = [
        ('year',        60*60*24*365),
        ('month',       60*60*24*30),
        ('day',         60*60*24),
        ('hour',        60*60),
        ('minute',      60),
        ('second',      1)
    ]

    for period_name, period_seconds in periods:
        if seconds > period_seconds:
            number = seconds / period_seconds
            return f"{number:.1f}{period_name[0]}"

    return "0s"