from functools import lru_cache

@lru_cache(maxsize=1)
def normalized_decay(decay, n):
    _decay = (decay - (decay ** (n + 1))) / (1 - (decay ** (n + 1)))
    return _decay, 1 - _decay

def ema_avg(averaged_model_parameter, model_parameter, num_averaged, decay):
    """
    Increment an exponential moving average.
    The approximate intuitive formula is:
    ema[n] = decay * ema[n-1] + (1 - decay) * parameter[n]

    However for small n the above is not normalized correctly; it is:
        ema[n] = (1 - decay) sum_{i=0}^n (decay^(n-i) x_i)
    we want a normalized ema instead:
        ema = sum_{i=0}^n (decay^(n-i) x_i ) / sum_{i=0}^n (decay^(n-i))
    (these are equivalent for large n)

    Simplifying: 
        ema = (1 - decay)/(1 - decay^{n+1}) * sum_{i=0}^n (decay^(n-i) x_i)
    So:
        ema[n] = 1/(1 - decay^{n+1}) * [
            ema[n-1] * (1 - decay ^ n) * decay +
            x[n] (1-decay)
        ]
    Or:
        ema[n] = _decay * ema[n-1] + (1 - _decay) * parameter[n]
    Where:
        _decay = (decay - decay^{n+1}) / (1 - decay^{n+1})
    """
    weight_old, weight_new = normalized_decay(decay, num_averaged)
    result = weight_old * averaged_model_parameter + weight_new * model_parameter
    return result
