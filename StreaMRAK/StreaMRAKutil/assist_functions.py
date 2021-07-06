import random
from numpy import sqrt
import numpy as np

def draw_samples(domain, target, n_samples):
    n, d = domain.shape
    if n_samples < n:
        idx = np.random.randint(0, n, size=n_samples)
    else:
        print(f'Number of samples: {n_samples}, is too large')
    return domain[idx, :], target[idx]


def _rand(data, seed):
    random.seed(seed)
    return random.randint(0, data.shape[0]-1)

def dist(v1,v2):
    """
    :param v1: 1-dim np array
    :param v2: 1-dim np array
    :return: distance between v1 and v2
    """
    return sqrt(sum((v1-v2)**2))

def estimate_span_of_data(domain, target):
    """Estimate the maximum distance between points in the dataset"""
    n, d = domain.shape
    if n <= 30000:
        n_samples = int(0.5*n)  # For very small datasets, we loop over 50% of the samples
    else:
        n_samples = 15000

    repeat = 10
    Dmax = []
    for j in range(repeat):
        X, _ = draw_samples(domain, target, n_samples)

        D = []
        for i in range(3):
            seed = 40 * i
            a = _rand(X, seed)
            seed = 44 * i
            b = _rand(X, seed)

            v1 = np.array(X[a, :])
            v2 = np.array(X[b, :])
            D.append(dist(v1, v2))
        Dmax.append(max(D))
    margin = 1.2 #margin to the estimate, to ensure that we include all points
    return margin * max(Dmax)

