import numpy as np

class GMM(object):

    def __init__(self, K, d):
        self.K = K
        self.d = d
        self.pi = np.random.rand(K)
        self.mu = np.random.randn(K, d)
        self.sigma = np.abs(np.random.randn(K, d, d))


    def fit(self, X, T):
        for t in range(T):
            print 'iter: {0}'.format(t)
            gamma = self.expectate(X)
            pi_hat, mu_hat, sigma_hat = self.maximize(X, self.mu, gamma)
            self.update_params(pi_hat, mu_hat, sigma_hat)


    def expectate(self, X):
        likel_funcs = self.__likelihood_functions(self.mu, self.sigma, self.K)
        likelihood = np.zeros((X.shape[0], self.K))
        for(i, x) in enumerate(X):
            likelihood[i, :] = [likel_funcs[k](x) for k in range(self.K)]
        _gamma = self.pi * likelihood
        return _gamma / _gamma.sum(axis = 1, keepdims = True)


    def maximize(self, X, mu, gamma):
        N_hat = gamma.sum(axis = 0)
        pi_hat = self.__estimate_pi(X.shape[0], N_hat)
        mu_hat = self.__estimate_mu(X, gamma, N_hat)
        sigma_hat = self.__estimate_sigma(X, mu, gamma, N_hat)
        return pi_hat, mu_hat, sigma_hat


    def update_params(self, pi_hat, mu_hat, sigma_hat):
        self.pi = pi_hat
        self.mu = mu_hat
        self.sigma = sigma_hat


    def __likelihood_functions(self, mu, sigma, K):
        return [self.__likelihood_function(mu[k, :], sigma[k, :, :]) for k in range(K)]


    def __likelihood_function(self, mu_k, sigma_k):
        def gauss(x):
            _x = x - mu_k
            numer = np.exp(-0.5 * _x.dot(np.linalg.inv(sigma_k).dot(_x.T)))
            denom = (2 * np.pi) ** (1.0 / self.d) * np.sqrt(np.abs(np.linalg.det(sigma_k)))
            return numer / denom
        return gauss


    def __estimate_pi(self, N, N_hat):
        return N_hat / N


    def __estimate_mu(self, X, gamma, N_hat):
        return gamma.T.dot(X) / N_hat.reshape(-1, 1)


    def __estimate_sigma(self, X, mu, gamma, N_hat):
        K, d = mu.shape
        _X = np.tile(X[:, None, :], (1, K, 1)) - mu
        _XX =  np.tile(_X[:, :, None, :], (1, 1, d, 1)) * _X[:, :, :, None]
        return (gamma[:, :, None, None] * _XX).sum(axis = 0) / N_hat[:, None, None]
