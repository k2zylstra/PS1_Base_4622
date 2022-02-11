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

class NaiveBayes(object):
    """
    NaiveBayes classifier for binary features and binary labels
    """

    def __init__(self, alpha=0.0):
        self.alpha = alpha
        self.classes_counts = None
        self.classes_log_probability = np.empty((2,))
        self.features_log_likelihood = []  # list of arrays where element i store log p(X[:,i], y)

    def compute_classes(self, y):
        """
        Computes the log prior of binary classes and stores the result in self.classes_log_probability
        :param y: binary labels array, shape (m,)
        """
        # Workspace 3.3
        #BEGIN 
        # code here
        positive_c = 0
        negative_c = 0
        size = len(y)
        for i in range(len(y)):
            if y[i]:
                positive_c += 1
            else:
                negative_c += 1
        
        self.classes_log_probability[0] = np.log(negative_c/size)
        self.classes_log_probability[1] = np.log(positive_c/size)
        #END

    def compute_features(self, X, y):
        """
        Computes the log likelihood matrices for different features and stores them in self.features_log_likelihood
        :param X: data matrix with binary features, shape (n_samples, n_features)
        :param y: binary labels array, shape (n_samples,)
        """
        # Workspace 3.4
        #BEGIN 
        # code here
        count_y_neg = 0
        count_y_pos = 0
        for i in range(len(y)):
            if y[i] == 0:
                count_y_neg += 1
            else:
                count_y_pos += 1
        for i in range(X.shape[1]):
            A_i = np.zeros((2,2))
            for j in range(X.shape[0]):
                if X[j][i] == 0 and y[j] == 0:
                    A_i[0][0] += 1
                elif X[j][i] == 0 and y[j] == 1:
                    A_i[1][0] += 1
                elif X[j][i] == 1 and y[j] == 0:
                    A_i[0][1] += 1
                elif X[j][i] == 1 and y[j] == 1:
                    A_i[1][1] += 1
            A_i[0][0] = A_i[0][0] / count_y_neg
            A_i[0][1] = A_i[0][1] / count_y_neg
            A_i[1][0] = A_i[1][0] / count_y_pos
            A_i[1][1] = A_i[1][1] / count_y_pos
            for j in range(A_i.shape[0]):
                for k in range(A_i.shape[1]):
                    A_i[j][k] = np.log(A_i[j][k])
            self.features_log_likelihood.append(A_i)
                
        #END

    def fit(self, X, y):
        """
        :param X: binary np.array of shape (n_samples, n_features) [values 0 or 1]
        :param y: corresponding binary labels of shape (n_samples,) [values 0 or 1]
        :return: Classifier
        """
        self.compute_classes(y)
        self.compute_features(X, y)
        return self

    def joint_log_likelihood(self, X):
        """
        Computes the joint log likelihood
        :param X: binary np.array of shape (n_samples, n_features) [values 0 or 1]
        :return: joint log likelihood array jll of shape (n_samples, 2), where jll[i] = [log p(X[i]|y=0),log p(X[i]|y=1)]
        """
        # Workspace 3.5
        #BEGIN 
        joint_log_likelihood = np.zeros((X.shape[0], 2))
        # code here

        for i in range(X.shape[0]):
            y_neg_case = 0
            y_pos_case = 0
            for j in range(X.shape[1]):
                if X[i][j] == 0:
                    y_neg_case += self.features_log_likelihood[j][0][0]
                    y_pos_case += self.features_log_likelihood[j][1][0]
                else: # X[i][j] == 1
                    y_neg_case += self.features_log_likelihood[j][0][1]
                    y_pos_case += self.features_log_likelihood[j][1][1]
            joint_log_likelihood[i][0] = y_neg_case
            joint_log_likelihood[i][1] = y_pos_case

        #END
        return joint_log_likelihood

    def predict(self, X):
        """
        :param X:
        :return:
        """

        # Workspace 3.6
        # TODO: Find the corresponding labels using Naive bayes logic
        #BEGIN 
        # code here
        y_hat = np.zeros((X.shape[0],))
        joint = self.joint_log_likelihood(X)
        for i in range(joint.shape[0]):
            p_y = (joint[i][1]+self.classes_log_probability[1])
            p_not_y = (joint[i][0]+self.classes_log_probability[0])
            if p_y > p_not_y:
                y_hat[i] = 1
            else:
                y_hat[i] = 0
        #END
        return y_hat

tests.test_NB(NaiveBayes)