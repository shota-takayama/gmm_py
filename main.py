import numpy as np
import numpy.random as nprd
from model.gmm import GMM

def stack(ndarr1, ndarr2):
    return np.vstack((ndarr1, ndarr2))

def create_data(K, d, N):
    pi = N * 1.0 / N.sum()
    mu = nprd.randn(K, d)
    sigma = np.array([s.dot(s.T) for s in nprd.randn(K, d, d)])
    X = reduce(stack, [nprd.multivariate_normal(mu[k], sigma[k], N[k]) for k in range(K)])
    return X, pi, mu, sigma

if __name__ == '__main__':

    # init
    K = 2
    d = 3
    N = nprd.randint(500, 3000, K)
    X, pi, mu, sigma = create_data(K, d, N)

    # train
    gmm = GMM(K, d)
    gmm.fit(X, T = 50)

    # result
    print '{}\n{}\n{}\n{}'.format('pi', pi, 'pi_est', gmm.pi)
    print '{}\n{}\n{}\n{}'.format('mu', mu, 'mu_est', gmm.mu)
    print '{}\n{}\n{}\n{}'.format('sigma', sigma, 'sigma_est', gmm.sigma)
