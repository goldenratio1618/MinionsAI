from minionsai.multiprocessing_rl.td_lambda import smooth_labels

import numpy as np

def test_smooth_labels_2():
    array = [1, 2]
    desired = np.array([2/3 * 1 + 1/3 * 2, 2.])
    np.testing.assert_allclose(smooth_labels(array, 0.5), desired)

def test_smooth_labels_3():
    array = [1, 2, 3]
    np.testing.assert_allclose(smooth_labels(array, 0.5), [4/7 * 1 + 2/7 * 2 + 1/7 * 3, 2/3 * 2 + 1/3 * 3, 3])