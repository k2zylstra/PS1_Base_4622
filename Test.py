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
        classes = np.unique(self._y)
        voted_labels = np.empty(indices_nearest_k.shape[0])
        for i in range(indices_nearest_k.shape[0]):
            dist_row = distances_nearest_k[i].copy()
            ind_row = indices_nearest_k[i].copy()

            while True:
                class_count = np.zeros((len(classes),2))
                for j in range(len(ind_row)):
                    c = self._y[ind_row[j]]
                    class_count[self.label_to_index[c]][0] += 1
                    class_count[self.label_to_index[c]][1] = c
                class_count = class_count[class_count[:, 0].argsort()]
                if len(ind_row) == 1:
                    voted_labels[i] = class_count[-1][1]
                    break
                if class_count[-1][0] != class_count[-2][0]:
                    voted_labels[i] = class_count[-1][1]
                    break
                farthest_index = 1
                farthest_dist = 0
                for j in range(1, len(dist_row)):
                    if dist_row[j] > farthest_dist:
                        farthest_index = j
                        farthest_dist = dist_row[j]
                dist_row = np.delete(dist_row, farthest_index)
                ind_row = np.delete(ind_row, farthest_index)
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
        #tree = sklearn.neighbors.BallTree(X, leaf_size=4)
        #self._ball_tree = sklearn.neighbors.BallTree(X)  # See documentation of BallTree and how it's used
        distances_nearest_k, indices_nearest_k = self._ball_tree.query(X, self._k)
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
        y_prime = self.predict(X)
        for i in range(len(y)):
            index1 = self.label_to_index[y[i]]
            index2 = self.label_to_index[y_prime[i]]
            c_matrix[index1][index2] += 1
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
        c_matrix = self.confusion_matrix(X, y)
        numer = 0
        denom = 0
        for i in range(c_matrix.shape[0]):
            numer += c_matrix[i][i]
        for i in range(len(c_matrix)):
            for j in range(len(c_matrix)):
                denom += c_matrix[i][j]
        score = numer / denom
        #END
        return score


class Numbers:
    def __init__(self):
        self.data = data.DigitData() # it has the same structure as binary_data

    def report(self):
        """
        Report information about the dataset using the print() function
        """
        # Workspace 2.1
        #TODO: Create print-outs for reporting the size of each set and the size of each datapoint
        #BEGIN 
        # code here
        #print("examples")
        #print("number of different partitions")
        #print("number of pixels")
        X_train = self.data.X_train
        X_valid = self.data.X_valid
        X_test = self.data.X_test
        print("number of different partitions:", 3)
        print("training partition number of examples:", X_train.shape[0])
        print("valid partition number of examples:", X_valid.shape[0])
        print("test partition number of examples:", X_test.shape[0])
        print("number of pixels per image:", X_train.shape[1])
        #END

    def evaluate(self, classifier_class):
        """
        valuates instances of the classifier class for different values of k and performs model selection
        :param classifier_class: Classifier class (either KNNClassifier or WeightedKNNClassifier)
        """

        # Workspace 2.2

        ks = list(range(1, 20))
        accuracies_valid = []
        #BEGIN 
        # code here (anything between BEGIN and END is yours to edit if needed)
        best_valid_k = None
        confusion_matrix = None
        accuracy = 0
        ks = list(range(1, 20))
        accuracies_valid = []
        for k in ks:
            print(k, end="\r")
            knn = classifier_class(k).fit(self.data.X_train, self.data.y_train)
            acc = knn.accuracy(self.data.X_valid, self.data.y_valid)
            accuracies_valid.append(accuracy)
            if accuracy < acc:
                accuracy = acc
                best_valid_k = k
        knn = classifier_class(best_valid_k).fit(self.data.X_train, self.data.y_train)
        confusion_matrix = knn.confusion_matrix(self.data.X_valid, self.data.y_valid)

        #END
        print("best k:", best_valid_k)
        print("Accuracy on test set:", accuracy)
        self.display_confusion(confusion_matrix)

    def view_digit(self, index, partition):
        """
        Display a digit given its index and partition
        :param index: index of the digit image
        :param partition: partition from which the digit is retrieved, either "train", "valid" or "test"
        """
        image = {"train": self.data.X_train, "valid": self.data.X_valid, "test": self.data.X_test}[partition][index]
        label = {"train": self.data.y_train, "valid": self.data.y_valid, "test": self.data.y_test}[partition][index]
        image = image.reshape(28, 28)
        plt.figure()
        plt.matshow(image)
        plt.title("Digit %i" % label)
        plt.show()

    @staticmethod
    def display_confusion(c_matrix):
        """
        Displays the confusion matrix using matshow
        :param c_matrix: square confusion matrix, shape (num_classes, num_classes)
        """
        _, ax = plt.subplots()
        ax.matshow(c_matrix, cmap=plt.cm.Blues)
        for i in range(c_matrix.shape[0]):
            for j in range(c_matrix.shape[0]):
                ax.text(i, j, str(c_matrix[j, i]), va='center', ha='center')
        plt.show()

numbers = Numbers()
numbers.report()
#numbers.evaluate(KNNClassifier)


numbers.view_digit(0, "train")