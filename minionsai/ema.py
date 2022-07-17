def ema_avg(averaged_model_parameter, model_parameter, num_averaged, decay):
    """
    We want a normalzied ema:
        ema = sum_{i=0}^n (lambda ^ (n-i) x_i ) / sum_{i=0}^n (lambda ^ (n-i))
    which is 
        ema = (1 - lambda)/(1 - lambda^{n+1}) * sum_{i=0}^n (lambda ^ (n-i) x_i)
    So:
        ema[n] = 1/(1 - lambda^{n+1}) * [
            ema[n-1] * (1 - lambda ^ n) * lambda +
            x[n] (1-lambda)
        ]
    """
    unnormalized_averaged = (1 - decay ** num_averaged) * averaged_model_parameter
    new_unnormalized_averaged = unnormalized_averaged * decay + model_parameter * (1 - decay)
    normalized_averaged = new_unnormalized_averaged / (1 - decay ** (num_averaged + 1))
    return normalized_averaged