
from typing import Callable
import random
import tqdm

import trueskill
from .minionsai.agent import Agent
from .minionsai.run_game import run_game
from .minionsai.engine import Game
import os

class TrueskillWorker():
    """
    Runs a trueskill environment for all agents in a directory.
    Watches the directory for new agents.
    """
    def __init__(self, directory: str, game_fn: Callable[[], Game], ratings: dict = None):
        self.directory = directory
        self.game_fn = game_fn
        self.agent_names = [] if ratings is None else list(ratings.keys())
        self.ratings = ratings or {}

    def run(self):
        while True:
            self.discover_agents()
            self.play_games()
            self.save_ratings()

    def discover_agents(self):
        """ List my directory and make sure I have all the agents."""
        print("Checking for new agents...")
        directory_agents = [agent_name for agent_name in os.listdir(self.directory) 
                            if not os.path.isdir(os.path.join(self.directory, agent_name))]
        for agent_name in set(self.agent_names) - set(directory_agents):
            print("Agent", agent_name, "is gone from directory.")
            self.agent_names.remove(agent_name)

        for agent_name in set(directory_agents) - set(self.agent_names):
            print(f"Found new agent: {agent_name}")
            self.agent_names.append(agent_name)
            self.ratings[agent_name] = trueskill.Rating()

    def load_agent(self, agent_name: str) -> Agent:
        return Agent.deserialize(os.path.join(self.directory, agent_name))

    def save_ratings(self):
        # TODO
        pass

    def play_games(self):
        agent_names = random.choice(self.agent_names)
        print(f"Playing games between {agent_names}")
        try:
            agent_0 = self.load_agent(agent_names[0])
            agent_1 = self.load_agent(agent_names[1])
        except Exception as e:
            print(f"Error loading agents:")
            print(e)
            return

        num_to_play = 100 # TODO make this dynamic based on ts
        for _ in tqdm.tqdm(range(num_to_play)):
            game = self.game_fn()
            winner = run_game(game, agents=[agent_0, agent_1], verbose=False)
            winner_rating = self.ratings[agent_names[winner]]
            loser_rating = self.ratings[agent_names[1 - winner]]
            new_winner_rating, new_loser_rating = trueskill.rate_1vs1(winner_rating, loser_rating)
            self.ratings[agent_names[winner]] = new_winner_rating
            self.ratings[agent_names[1 - winner]] = new_loser_rating
        print(f"{agent_names[winner]}: {new_winner_rating.mu:.1f} +/- {new_winner_rating.sigma:.1f}")
        print(f"{agent_names[1 - winner]}: {new_loser_rating.mu:.1f} +/- {new_loser_rating.sigma:.1f}")