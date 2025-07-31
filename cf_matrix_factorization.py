
import numpy as np
from sklearn.metrics import mean_squared_error

class MatrixFactorization:
    def __init__(self, R, K, alpha, beta, iterations):
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        self.training_process = []
        for epoch in range(self.iterations):  # use a unique name like 'epoch'
            for u in range(self.num_users):
                for j in range(self.num_items):  # fix this from 'i' to 'j'
                    if self.R[u, j] > 0:
                        eui = self.R[u, j] - np.dot(self.P[u, :], self.Q[j, :].T)
                        self.P[u, :] += self.alpha * (eui * self.Q[j, :] - self.beta * self.P[u, :])
                        self.Q[j, :] += self.alpha * (eui * self.P[u, :] - self.beta * self.Q[j, :])
            rmse = self.rmse()
            print(f"Iteration: {epoch+1}, RMSE: {rmse:.4f}")

    def rmse(self):
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error / len(xs))

    def full_matrix(self):
        return np.dot(self.P, self.Q.T)
