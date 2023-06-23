import numpy as np
from numpy.linalg import norm
from random import normalvariate
from math import sqrt

def random_unit_vector(size):
    unnormalized = [normalvariate(0, 1) for _ in range(size)]
    norm = sqrt(sum(v * v for v in unnormalized))
    return [v / norm for v in unnormalized]


def power_iterate(X, epsilon=1):
    """ Recursively compute X^T X dot v to compute weights vector/eignevector """

    n, m= X.shape
    start_v = random_unit_vector(m)  # start of random surf
    curr_eigenvector = start_v
    covariance_matrix = np.dot(X.T, X)

    ## power iterationn until converges
    it = 0
    while True:
        it += 1
        prev_eigenvector = curr_eigenvector
        curr_eigenvector = np.dot(covariance_matrix, prev_eigenvector)
        curr_eigenvector = curr_eigenvector / norm(curr_eigenvector)

        if abs(np.dot(curr_eigenvector, prev_eigenvector)) > 1 - epsilon:
            return curr_eigenvector

def svd(X, epsilon=1e-100):

    """after computed change of basis matrix from power iteration, compute distance"""
    n, m= X.shape
    change_of_basis = []

    for i in range(m):
        data_matrix = X.copy()

        for sigma, u, v in change_of_basis[:i]:
            data_matrix -= sigma * np.outer(u, v)

        v = power_iterate(data_matrix, epsilon=epsilon)  ## eigenvector
        u_sigma = np.dot(X, v)  ## 2nd step: XV = U Sigma
        sigma = norm(u_sigma)
        u = u_sigma / sigma

        change_of_basis.append((sigma, u, v))

    sigmas, us, v_transposes = [np.array(x) for x in zip(*change_of_basis)]

    return np.dot(np.dot(us.T ,np.diag(sigmas)),v_transposes)