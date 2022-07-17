from minionsai.ema import ema_avg
import torch as th

def test_ema_normalized():
    data = th.tensor([1.0])
    DECAY = 0.2  # Doesn't matter
    for i in range(1, 10):
        data = ema_avg(data, th.tensor([1.0]), i, DECAY)
        th.testing.assert_allclose(data.item(), 1.0)

def test_ema_zero_decay():
    # At decay 0 we just take the last entry
    data = th.rand(1)
    DECAY = 0.0
    for i in range(1, 10):
        print(data)
        data = ema_avg(data, th.rand_like(data), i, DECAY)
    data = ema_avg(data, th.tensor([7.0]), 10, DECAY)
    th.testing.assert_allclose(data.item(), 7.0)

def test_ema_decay():
    start = th.tensor([16.0])
    data = th.clone(start)
    DECAY = 0.5
    iters = 4
    for i in range(1, iters + 1):
        data = ema_avg(data, th.zeros_like(data), i, DECAY)
        print(data)
    expected_result = start * DECAY ** iters * (1 - DECAY) / (1 - DECAY ** (iters + 1))
    th.testing.assert_allclose(data.item(), expected_result.item())