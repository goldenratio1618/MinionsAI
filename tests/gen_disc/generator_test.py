from collections import defaultdict
from minionsai.action import ActionList, MoveAction, SpawnAction
from minionsai.action_bot.model import MinionsActionBot
from minionsai.discriminator_only.translator import Translator
from minionsai.engine import Game
from minionsai.game_util import seed_everything
from minionsai.gen_disc.generator import QGenerator, argmax_last_two_indices, gumbel_sample
from minionsai.gen_disc.tree_search import DepthFirstTreeSearch
from minionsai.unit_type import ZOMBIE
import numpy as np
import torch as th

def test_argmax_last_two_indices():
    array = np.array([
        [[1, 2, 3], [4, 5, 6]],
        [[4.01, 4.02, 4], [-1, 0, 3]]])
    output = argmax_last_two_indices(array)
    np.testing.assert_equal(output, [[1, 2], [0, 1]])

def test_gumbel_sample_t_0():
    array = np.array([
        [[-1, 1, 10]],
        [[4.01, 4.02, 4]]])
    for _ in range(10):
        output = gumbel_sample(array, temperature=0.0)
        # Should be the argmax every time.
        np.testing.assert_equal(output, [[0, 2], [0, 1]])

def test_gumbel_sample_t_large():
    array = np.array([
        [[-1], [1], [10]],
        [[4.01], [4.02], [4]]])
    counts = defaultdict(int)
    for _ in range(100):
        output = gumbel_sample(array, temperature=1000)
        counts[tuple(output.flatten())] += 1
    # Check that we got every possible outcome:
    for i in range(3):
        for j in range(3):
            assert counts[(i, 0, j, 0)] > 0

def test_gumbel_sample_t_medium():
    array = np.array([
        [[-1], [1], [10]],
        [[4.01], [4.02], [4]]])
    counts = defaultdict(int)
    for _ in range(100):
        output = gumbel_sample(array, temperature=0.02)
        counts[tuple(output.flatten())] += 1
    # Should always sample 10 in the first row:
    for j in range(3):
        assert counts[(0, 0, j, 0)] == 0
        assert counts[(1, 0, j, 0)] == 0
    # Check that we got every possible outcome in the second row:
    for j in range(3):
        assert counts[(2, 0, j, 0)] > 0
    # Check we got more of the more likely thing:
    assert counts[(2, 0, 1, 0)] > counts[(2, 0, 2, 0)]

def test_gumbel_sample_masks():
    array = np.array([
        [[-1], [1], [10]],
        [[-4.2], [-np.inf], [-4]]])
    counts = defaultdict(int)
    for _ in range(100):
        output = gumbel_sample(array, temperature=np.random.random())
        counts[tuple(output.flatten())] += 1
    # Check that we never sampled the inf in teh second row
    for i in range(3):
        assert counts[(i, 0, 1, 0)] == 0

class MockModelUpperLeft():
    # mock model prefers actions near the upper-right corner of the actions grid.
    def __call__(self, obs):
        return -th.arange(30**2).view(1, 30, 30)  # Assumes num_things = 30

class MockModelRandom():
    # mock model prefers actions near the upper-right corner of the actions grid.
    def __call__(self, obs):
        return th.rand(1, 30, 30)  # Assumes num_things = 30

def test_qgenerator_tree_search_first_turn_1():
    seed_everything(3)
    game = Game()
    game.next_turn()
    model=MockModelUpperLeft()
    generator = QGenerator(translator=Translator(mode='generator'), model=model, sampling_temperature=0, epsilon_greedy=0)
    action_lists, final_game_states, training_datas = generator.tree_search(game, num_trajectories=2)

    assert len(action_lists) == 2
    # Because we're following mockmodel, we should have taken actions: (0 1), (25 25) (25 0) (25 25)
    seed_everything(3)
    recreated_game = Game()
    recreated_game.next_turn()
    recreated_game.full_turn(ActionList([MoveAction((0, 0), (0, 1))], [SpawnAction(ZOMBIE, (0, 0))]))
    assert hash(final_game_states[0]) == hash(recreated_game), final_game_states[0].pretty_print() + "\n\n" + recreated_game.pretty_print()

    # Because we're following mockmodel, next best action should be: (0 5), (25 25) (25 0) (25 25)
    seed_everything(3)
    recreated_game = Game()
    recreated_game.next_turn()
    recreated_game.full_turn(ActionList([MoveAction((0, 0), (1, 0))], [SpawnAction(ZOMBIE, (0, 0))]))
    assert hash(final_game_states[1]) == hash(recreated_game), final_game_states[1].pretty_print() + "\n\n" + recreated_game.pretty_print()

def test_qgenerator_tree_search_first_turn_all():
    game = Game(money=(2, 4))
    game.next_turn()
    model=MockModelUpperLeft()
    generator = QGenerator(translator=Translator(mode='generator'), model=model, sampling_temperature=0, epsilon_greedy=0)
    action_lists, final_game_states, training_datas = generator.tree_search(game, num_trajectories=20)
    # At the first turn, there are 13 possible actions (Necro in corner with 2 zombie spots; necro on either side with 4 zombie spots each; necro in any of the 3 spots with no zombies)
    assert len(action_lists) == 13
    assert len(final_game_states) == 13
    assert len(training_datas) == 13