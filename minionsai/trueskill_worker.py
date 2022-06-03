
from typing import Callable
import random
from sqlalchemy import true
import tqdm
import trueskill
import os
from tabulate import tabulate
import csv

from .agent import Agent
from .run_game import run_game
from .engine import Game


class TrueskillWorker():
    """
    Runs a trueskill environment for all agents in a directory.
    Watches the directory for new agents.
    """
    def __init__(self, directory: str, game_fn: Callable[[], Game], batch_size: int=1, ratings: dict = {}, num_games: dict = {}):
        self.directory = directory
        self.game_fn = game_fn
        self.agent_names = [] if ratings is None else list(ratings.keys())
        self.ratings = ratings or {}
        self.num_games = num_games or {}
        self.choose_underevaluated_agents_prob = 0.5
        self.batch_size = batch_size
        self.env = trueskill.TrueSkill(tau=0, draw_probability=0.0)

    def run(self):
        while True:
            self.discover_agents()
            self.play_games()
            self.save_ratings()

    def discover_agents(self):
        """ List my directory and make sure I have all the agents."""
        print("Checking for new agents...")
        directory_agents = [agent_name for agent_name in os.listdir(self.directory) 
                            if os.path.isdir(os.path.join(self.directory, agent_name))]

        for agent_name in set(self.agent_names) - set(directory_agents):
            print("Agent", agent_name, "is gone from directory.")
            self.agent_names.remove(agent_name)

        for agent_name in set(directory_agents) - set(self.agent_names):
            print(f"Found new agent: {agent_name}")
            self.agent_names.append(agent_name)
            self.ratings[agent_name] = trueskill.Rating()
            self.num_games[agent_name] = 0

    def load_agent(self, agent_name: str) -> Agent:
        return Agent.load(os.path.join(self.directory, agent_name))

    def save_ratings(self):
        # Save a csv with columns ['name', 'trueskill', 'trueskill_sigma', 'games_played']
        with open(os.path.join(self.directory, "scores.csv"), "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["name", "trueskill", "trueskill_sigma", "games_played"])
            for name in self.agent_names:
                writer.writerow([name, self.ratings[name].mu, self.ratings[name].sigma, self.num_games[name]])

    def print_ratings(self):
        table = []
        for name in self.agent_names:
            table.append([name, self.ratings[name].mu, self.ratings[name].sigma, self.num_games[name]])
        table.sort(key=lambda x: x[1], reverse=True)
        #Print with 2 decimal places
        return tabulate(table, headers=["Name", "Mu", "Sigma", "Games"], floatfmt=".1f")

    def choose_agents(self):
        if random.random() < self.choose_underevaluated_agents_prob:
            # Make sure to choose one agent with minimal num_games
            min_num_games = min(self.num_games.values())
            min_games_agent_names = [name for name in self.num_games if self.num_games[name] == min_num_games]
            first_agent = random.choice(min_games_agent_names)
            other_agents = [name for name in self.agent_names if name != first_agent]
            second_agent = random.choice(other_agents)
            return [first_agent, second_agent]
        else:
            return random.sample(self.agent_names, 2)

    def play_games(self):
        agent_names = self.choose_agents()
        print(f"Playing games between {agent_names}")
        try:
            agent_0 = self.load_agent(agent_names[0])
            agent_1 = self.load_agent(agent_names[1])
            agents = [agent_0, agent_1]
        except Exception as e:
            print(f"Error loading agents:")
            print(e)
            return

        wins = {name: 0 for name in agent_names}
        for i in tqdm.tqdm(range(self.batch_size)):
            game = self.game_fn()
            player0 = i % 2
            agents_shuffled = [agents[player0], agents[1 - player0]]
            agent_names_shuffled = [agent_names[player0], agent_names[1 - player0]]
            winner = run_game(game, agents=agents_shuffled, verbose=False)
            winner_name = agent_names_shuffled[winner]
            loser_name = agent_names_shuffled[1 - winner]
            winner_rating = self.ratings[winner_name]
            loser_rating = self.ratings[loser_name]
            new_winner_rating, new_loser_rating = trueskill.rate_1vs1(winner_rating, loser_rating, env=self.env)
            self.ratings[winner_name] = new_winner_rating
            self.ratings[loser_name] = new_loser_rating
            self.num_games[winner_name] += 1
            self.num_games[loser_name] += 1
            wins[winner_name] += 1
        for name in agent_names:
            print(f"{name} wins {wins[name]/self.batch_size:.1%}; new rating {self.ratings[name].mu:.1f} +/- {self.ratings[name].sigma:.1f}    num_games={self.num_games[name]}")

        print(self.print_ratings())