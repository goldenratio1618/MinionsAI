
import datetime
import traceback
from typing import Callable, Optional
import random
import tqdm
import trueskill
import os
from tabulate import tabulate
import csv

from .agent import Agent
from .agent_saveload import load
from .run_game import AgentException, run_game
from .engine import Game

def read_scores(scores_file):
    # scores are stored in scores.csv
    # With columns 'name', 'trueskill', 'trueskill_sigma', 'games_played'

    # If the file doesn't exist, return an empty list
    if not os.path.isfile(scores_file):
        print(f"No scores file found at {scores_file}")
        return [], None
    with open(scores_file, 'r') as f:
        reader = csv.reader(f)
        # first line is header; check that it matches our expectations
        header = next(reader)
        if header != ['name', 'trueskill', 'trueskill_sigma', 'games_played', 'ok', 'crashes']:
            raise Exception(f"Unexpected header in {scores_file}: {header}")
        scores = list(reader)
    scores = [{'name': s[0], 'trueskill': float(s[1]), 'trueskill_sigma': float(s[2]), 'games_played': int(s[3]), 'ok': s[4], 'crashes': int(s[5])} for s in scores]
    last_update = datetime.datetime.fromtimestamp(os.path.getmtime(scores_file))
    return scores, last_update

class TrueskillWorker():
    """
    Runs a trueskill environment for all agents in a directory.
    Watches the directory for new agents.
    """
    def __init__(self, directory: str, game_fn: Callable[[], Game], batch_size: int=1, restart=False, scores_file=None):
        self.directory = directory
        self.game_fn = game_fn
        self.choose_underevaluated_agents_prob = 0.5
        self.batch_size = batch_size
        self.env = trueskill.TrueSkill(tau=0.01, draw_probability=0.0)
        self.dropped_agents = []
        self.max_crash_rate = 0.5
        self.scores_file = scores_file or os.path.join(directory, "scores.csv")

        # Restart from from where the last thread stopped if possible.
        prev_scores, _ = read_scores(self.scores_file)
        if restart:
            # Restart even if there was previously data
            prev_scores = []

        if len(prev_scores) > 0:
            print(f"Restarting from previous scores found at {self.scores_file}")
        self.agent_names = [agent_dict['name'] for agent_dict in prev_scores]
        self.ratings = {agent_dict['name']: trueskill.Rating(agent_dict['trueskill'], agent_dict['trueskill_sigma']) 
                        for agent_dict in prev_scores}
        self.num_games = {agent_dict['name']: agent_dict['games_played'] for agent_dict in prev_scores}
        self.num_crashes = {agent_dict['name']: agent_dict['crashes'] for agent_dict in prev_scores}

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

        for agent_name in set(directory_agents) - set(self.agent_names) - set(self.dropped_agents):
            print(f"Found new agent: {agent_name}")
            self.agent_names.append(agent_name)
            self.ratings[agent_name] = trueskill.Rating()
            self.num_games[agent_name] = 0
            self.num_crashes[agent_name] = 0

    def load_agent(self, agent_name: str) -> Optional[Agent]:
        try:
            test_load_equivalence = self.num_games[agent_name] % 100 == 0
            return load(os.path.join(self.directory, agent_name), test_load_equivalence=test_load_equivalence)
        except Exception as e:
            print(f"Error loading agent: {agent_name}")
            print(e)

            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            stack_trace_file = os.path.join(self.directory, agent_name, f"stacktrace_load_{timestamp}.txt")
            print(f"Saving stack trace to: {stack_trace_file}")
            with open(stack_trace_file, "w") as f:
                traceback.print_exc(file=f)
            print("I'm going to stop worrying about that agent.")
            self.drop_agent(agent_name)
            self.num_crashes[agent_name] += 1
            return None

    def drop_agent(self, agent_name: str):
        self.dropped_agents.append(agent_name)
        self.agent_names.remove(agent_name)

    def save_ratings(self):
        # Save a csv with columns ['name', 'trueskill', 'trueskill_sigma', 'games_played']
        with open(self.scores_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["name", "trueskill", "trueskill_sigma", "games_played", "ok", "crashes"])
            for name in self.agent_names:
                writer.writerow([name, self.ratings[name].mu, self.ratings[name].sigma, self.num_games[name], True, self.num_crashes[name]])
            for name in self.dropped_agents:
                writer.writerow([name, self.ratings[name].mu, self.ratings[name].sigma, self.num_games[name], False, self.num_crashes[name]])

    def print_ratings(self):
        table = []
        for name in self.agent_names:
            table.append([name, self.ratings[name].mu, self.ratings[name].sigma, self.num_games[name], True, self.num_crashes[name]])
        for name in self.dropped_agents:
            table.append([name, self.ratings[name].mu, self.ratings[name].sigma, self.num_games[name], False, self.num_crashes[name]])
        table.sort(key=lambda x: x[1], reverse=True)
        return tabulate(table, headers=["Name", "Mu", "Sigma", "Games", "OK", 'Crashes'], floatfmt=".1f")

    def choose_agents(self):
        print(f"Choosing among agents {self.agent_names}")
        if random.random() < self.choose_underevaluated_agents_prob:
            # Make sure to choose one agent with minimal num_games
            min_num_games = min([self.num_games[a] for a in self.agent_names])
            min_games_agent_names = [name for name in self.agent_names if self.num_games[name] == min_num_games]
            first_agent = random.choice(min_games_agent_names)
            other_agents = [name for name in self.agent_names if name != first_agent]
            second_agent = random.choice(other_agents)
            return [first_agent, second_agent]
        else:
            return random.sample(self.agent_names, 2)

    def play_games(self):
        agent_names = self.choose_agents()
        print(f"Playing games between {agent_names}")
        agents = [self.load_agent(a) for a in agent_names]
        if None in agents:
            print("One or more agents failed to load. Skipping this round.")
            return

        wins = {name: 0 for name in agent_names}
        for i in tqdm.tqdm(range(self.batch_size)):
            game = self.game_fn()
            player0 = i % 2
            agents_shuffled = [agents[player0], agents[1 - player0]]
            agent_names_shuffled = [agent_names[player0], agent_names[1 - player0]]
            try:
                winner = run_game(game, agents=agents_shuffled, verbose=False)
            except AgentException as e:
                agent_index = e.agent_index
                offender = agent_names_shuffled[agent_index]
                print(f"Agent {offender} threw an exception during game {i}")
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                stack_trace_file = os.path.join(self.directory, offender, f"stacktrace_play_{timestamp}.txt")
                print(f"Saving stack trace to: {stack_trace_file}")
                with open(stack_trace_file, "w") as f:
                    traceback.print_exc(file=f)

                self.num_crashes[offender] += 1

                # If the agent crashed too many times, drop it
                enough_games = self.num_crashes[offender] >= self.batch_size * 3
                too_many_crashes =self.num_crashes[offender] / (self.num_games[offender] + self.num_crashes[offender]) > self.max_crash_rate
                print(f"Enough games: {enough_games}, too many crashes: {too_many_crashes}")
                if enough_games and too_many_crashes:
                    print(f"Agent {offender} has crashed too many times. Dropping it.")
                    self.drop_agent(offender)
                    break
                continue

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