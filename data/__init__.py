import numpy as np
import os

current_folder = os.path.dirname(os.path.abspath(__file__))
from sklearn.metrics.pairwise import euclidean_distances


class BinaryData(object):
    def __init__(self, sqrt_n_samples=24):
        np.random.seed(24)
        self.sqrt_n_samples = sqrt_n_samples
        self.n_samples = sqrt_n_samples ** 2
        one_d = np.linspace(-sqrt_n_samples / 20.0, sqrt_n_samples / 20.0, sqrt_n_samples)
        X = np.meshgrid(one_d, one_d)
        X = np.concatenate([X[0].reshape(-1, 1), X[1].reshape(-1, 1)], axis=1)

        X = X + np.random.uniform(-0.035, 0.035, (self.n_samples, 2))
        np.random.shuffle(X)
        y = ((np.cos(1.5 * X[:, 0]) - 0.5) < X[:, 1]) * 1.0
        flip = np.random.choice([0, 1], size=self.n_samples, p=[0.7, 0.3])
        close_to_boundary = np.cos(1.5 * X[:, 0]) - 0.5 - X[:, 1] < 0.3
        close_to_boundary = np.logical_and(0 < np.cos(1.5 * X[:, 0]) - 0.5 - X[:, 1], close_to_boundary)
        flip = np.logical_and(flip, close_to_boundary) * 1.0
        y = y * (1 - flip) + flip * (1 - y)
        distances = euclidean_distances(X, X)

        self.X, self.y = X, y
        self.one_to_zero = np.min(distances[y == 1][:, y == 0], axis=1)

        self.X_train, self.y_train = X[:self.n_samples // 3], y[:self.n_samples // 3]
        self.X_valid, self.y_valid = (X[self.n_samples // 3: 2 * self.n_samples // 3],
                                      y[self.n_samples // 3: 2 * self.n_samples // 3])
        self.X_test, self.y_test = X[2 * self.n_samples // 3:], y[2 * self.n_samples // 3:]

        a = self.y_train[self.y_train == 1]
        b = a[self.one_to_zero[:sum(self.y_train == 1)] < 0.3]
        b[np.random.choice([True, False], len(b), p=[0.4, 0.6])] = 0
        a[self.one_to_zero[:sum(self.y_train == 1)] < 0.3] = b
        self.y_train[np.where(self.y_train == 1)[0]] = a

    def boundary(self):
        x = np.sort(self.X[:, 0])
        return x, np.cos(1.5 * x) - 0.5


class DigitData(object):

    def __init__(self):
        loaded = np.load(os.path.join(current_folder, "mnist.npz"))
        self.images = images = loaded["images"].reshape(-1, 28 * 28)
        self.labels = labels = loaded["labels"]
        train_size = 1000
        valid_size = 500
        test_size = 500
        self.X_train, self.y_train = images[:train_size], labels[:train_size]
        self.X_valid, self.y_valid = images[train_size: train_size + valid_size], labels[
                                                                                  train_size: train_size + valid_size]
        self.X_test, self.y_test = (images[train_size + valid_size:train_size + valid_size + test_size],
                                    labels[train_size + valid_size: train_size + valid_size + test_size])
