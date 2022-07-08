from minionsai.discriminator_only.translator import Translator
from minionsai.engine import BOARD_SIZE
import numpy as np

def test_transpose_actions():
    assert BOARD_SIZE == 5, "This test assumes BOARD_SIZE == 5"
    actions = np.array([
        [21, 14],  # (4, 1) -> (1, 4) = 9; (2, 4) -> (4, 2) = 22
        [25, 1],   # special; (0, 1) -> (1, 0) = 5
         [0, 3]    # (0, 0) -> (0, 0) = 0; (0, 3) -> (3, 0) = 15
         ])
    flipped_actions = Translator.transpose_actions(actions)
    np.testing.assert_equal(flipped_actions, [[9, 22], [25, 5], [0, 15]])


def test_rotate_actions():
    assert BOARD_SIZE == 5, "This test assumes BOARD_SIZE == 5"
    actions = np.array([
        [21, 14],  # (4, 1) -> (0, 3) = 3; (2, 4) -> (2, 0) = 10
        [25, 1],   # special; (0, 1) -> (4, 3) = 23
         [0, 3]    # (0, 0) -> (4, 4) = 24; (0, 3) -> (4, 1) = 21
         ])
    rotated_actions = Translator.rotate_actions(actions)
    np.testing.assert_equal(rotated_actions, [[3, 10], [25, 23], [24, 21]])