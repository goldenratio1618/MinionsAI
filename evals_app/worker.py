""" Run this to evaluate agents in the database. """

import random
import time
from tkinter import N
import trueskill
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from models import Game, Agent, Env

# The db is at evals.db in the same dir as this script.
path = os.path.join(os.path.dirname(__file__), 'evals.db')
engine = create_engine(f"sqlite:///{path}")


def run(env_name):
    while True:
        read_session = Session(engine)
        env = read_session.query(Env).filter_by(name=env_name).first()
        print(f"Found env: {env.name}")

        (p1, p2) = setup_game(env)
        if p1 is None: # No requested games
            print("No requested games; sleeping 10s")
            time.sleep(10)
            continue

        game = env.build()

        try:
            # TODO figure out API
            agent1 = p1.build()
            agent2 = p2.build()
            p1_wins = game.run(agent1, agent2)
        except Exception as e:
            print("\n\n\n")
            print(f"Excpetion in game from env {env_name}")
            print(f"Agents were {p1.name} and {p2.name}")
            print("===============================")
            print(e)
            continue

        read_session.close()
        print("Game complete.")
        update(p1, p2, p1_wins, env)
        print("Update complete")

def setup_game(env):
    print([a.queued_games for a in env.agents])
    requested_agents = [a for a in env.agents if a.queued_games > 0]
    if len(requested_agents) == 0:
        return None, None
    p1 = random.choice(requested_agents)
    p2 = random.choice(env.agents)
    agents = [p1, p2]
    random.shuffle(agents)
    return agents

def update(p1, p2, p1_wins, env):
    session = Session(engine)
    # relaod p1 and p2 in this session
    p1 = session.query(Agent).filter_by(name=p1.name).first()
    p2 = session.query(Agent).filter_by(name=p2.name).first()

    if p1.queued_games > 0:
        p1.queued_games -= 1
    if p2.queued_games > 0:
        p2.queued_games -= 1

    game = Game(env_id=env.id, num_wins_p1=p1_wins, p1_id=p1.id, p2_id=p2.id)
    session.add(game)
    p1_rating = trueskill.Rating(mu=p1.trueskill, sigma=p1.trueskill_sigma)
    p2_rating = trueskill.Rating(mu=p2.trueskill, sigma=p2.trueskill_sigma)
    if p1_wins:
        p1_rating, p2_rating = trueskill.rate_1vs1(p1_rating, p2_rating)
    else:
        p2_rating, p1_rating = trueskill.rate_1vs1(p2_rating, p1_rating)
    p1.trueskill = p1_rating.mu
    p1.trueskill_sigma = p1_rating.sigma
    p2.trueskill = p2_rating.mu
    p2.trueskill_sigma = p2_rating.sigma
    session.commit()
    session.close()

if __name__ == "__main__":
    run("test")