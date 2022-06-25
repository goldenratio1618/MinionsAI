import numpy as np

def smooth_labels(labels, lam):
    """
    Takes a list of win probs with a 0/1 at the end
    and smooths them into targets for the model
    using exponential moving average
    """
    reversed_labels = np.array(labels)[::-1]
    lambda_powers = lam ** np.arange(len(labels), 0, -1)
    turn_contribs = np.cumsum(reversed_labels * lambda_powers)
    norm = np.cumsum(lambda_powers)
    return (turn_contribs / norm)[::-1]
