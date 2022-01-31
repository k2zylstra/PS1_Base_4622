from turtle import distance
from idna import InvalidCodepoint
import numpy as np
import matplotlib.pyplot as plt
import sklearn.neighbors
import data
import tests


class KNNClassifier:

    def __init__(self, k=5):
        """
        Initialize our custom KNN classifier
        :param k: the number of nearest neighbors to consider for classification
        """
        self._k = k
        self._ball_tree = None
        self._y = None
        self.label_to_index = None
        self.index_to_label = None

    def fit(self, X, y):
        """
        Fit the model using the provided data
        :param X: 2-D np.array of shape (number training samples, number of features)
        :param y: 1-D np.array of shape (number training samples,)
        :return: self
        """
        self._ball_tree = sklearn.neighbors.BallTree(X)  # See documentation of BallTree and how it's used
        self._y = y
        # Should be used to map the classes to {0,1,..C-1} if needed (C is the number of classes)
        # We can assume that the training data contains samples from all the possible classes
        classes = np.unique(y)
        self.label_to_index = dict(zip(classes, range(classes.shape[0])))
        self.index_to_label = dict(zip(range(classes.shape[0]), classes))

        return self

    def majority_vote(self, indices_nearest_k, distances_nearest_k=None):
        """
        Given indices of the nearest k neighbors for each point, report the majority label of those points.
        :param indices_nearest_k: np.array containing the indices of training neighbors, of shape (M, k)
        :param distances_nearest_k: np.array containing the corresponding distances of training neighbors, of shape (M, k)
        :return: The majority label for each row of indices, shape (M,)
        """

        # Workspace 1.1
        # TODO: Determine majority for each row of indices_nearest_k
        # TODO: if there is a tie, remove the farthest neighbor until the tie is broken
        #BEGIN 
#        voted_labels = np.empty(indices_nearest_k.shape[0])
#        for i in range(indices_nearest_k.shape[0]):
#            min_index = 0
#            min_dist = 0
#            for j in range(distances_nearest_k.shape[1]):
#                if distances_nearest_k[i][j] < min_dist:
#                    min_index = j
#                    min_dist = distances_nearest_k[i][j]
#            voted_labels[i] = self.index_to_label[min_index]
        # code here
        voted_labels = np.empty(indices_nearest_k.shape[0])
        for i in range(indices_nearest_k.shape[0]):
            class_count = np.empty(len(self.classes),2)
            while True:
                for j in range(indices_nearest_k.shape[1]):
                    c = self._y[j]
                    class_count[self.label_to_index[c]][0] += 1
                    class_count[self.label_to_index[c]][1] = c
                sort(class_count)
                if class_count[len(class_count)-1] != class_count[len(class_count)-2]:
                    voted_labels[i] = class_count[len(class_count)-1][1]
                    break
                farthest_index = 0
                farthest_dist = 0
                for j in range(distances_nearest_k.shape[1]):
                    if distances_nearest_k[i][j] > farthest_dist:
                        farthest_index = j
                        farthest_dist = distances_nearest_k[i][j]
                distances_nearest_k.remove(farthest_index)
                indices_nearest_k.remove(farthest_index)
        #END
        return voted_labels

    def predict(self, X):
        """
        Given new data points, classify them according to the training data provided in fit and number of neighbors k
        You should use BallTree to get the distances and indices of the nearest k neighbors
        :param X: feature vectors (num_points, num_features)
        :return: 1-D np.array of predicted classes of shape (num_points,)
        """
        # Workspace 1.2
        #BEGIN 
        # code here
        tree = sklearn.neighbors.BallTree(X, leaf_size=4)
        distances_nearest_k, indices_nearest_k = tree.query(X, self._k)
        #END
        return self.majority_vote(indices_nearest_k, distances_nearest_k)

    def confusion_matrix(self, X, y):
        """
        Generate the confusion matrix for the given data
        :param X: an np.array of feature vectors of points, shape (N, n_features)
        :param y: the corresponding correct classes of our set, shape (N,)
        :return: a C*C np.array of counts, where C is the number of classes in our training data
        """
        # The rows of the confusion matrix correspond to the counts from the true labels, the columns to the predictions'
        # Workspace 1.3
        # TODO: Run classification for the test set X, compare to test answers y, and add counts to matrix
        c_matrix = np.zeros((len(self.label_to_index), len(self.label_to_index)))
        #BEGIN 
        # code here
        #END
        return c_matrix

    def accuracy(self, X, y):
        """
        Return the accuracy of the classifier on the data (X_test, y_test)
        :param X: np.array of shape (m, number_features)
        :param y: np.array of shape (m,)
        :return: accuracy score [float in (0,1)]
        """
        # Workspace 1.4
        # TODO: Compute accuracy on X
        #BEGIN 
        # code here
        score = 0 # REPLACE
        #END
        return score

tests.testKNN(KNNClassifier)