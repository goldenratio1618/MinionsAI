from collections import defaultdict
from minionsai.gen_disc.generator import argmax_last_two_indices, gumbel_sample
import numpy as np

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