import matplotlib.pyplot as plt
import numpy as np


def show_decision_surface(model, X, y, ax=None):
    """
    Helper function to visualize the decision surface of the trained model
    :param model with predict method
    :return: None
    """
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    x_grid, y_grid = np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05)
    xx, yy = np.meshgrid(x_grid, y_grid)
    r1, r2 = xx.reshape(-1, 1), yy.reshape(-1, 1)
    grid = np.hstack((r1, r2))
    y_hat = model.predict(grid).reshape(-1, )
    zz = y_hat.reshape(xx.shape)

    if ax is None:
        plt.contourf(xx, yy, zz, cmap='PiYG')
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.show()
    else:
        ax.contourf(xx, yy, zz, cmap='PiYG')
        ax.scatter(X[:, 0], X[:, 1], c=y)


class Tester(object):
    def __init__(self):
        self.questions = {}

    def add_test(self, question, test_function):
        self.questions[question] = test_function

    def run(self):
        for question in self.questions:
            success, comment = self.questions[question]()
            if success:
                print("Question %s: [PASS]" % question)
            else:
                print("Question %s: [FAIL]" % question, comment)


def testKNN(knnclass):
    tester = Tester()
    features = np.array([[1, 1], [1, 2], [2, 1], [5, 2], [3, 2], [8, 2], [2, 4]])
    labels = np.array([1, -1, -1, 1, -1, 1, -1])
    test_points = np.array([[1, 1.1], [3, 1], [7, 5], [2, 6], [4, 4]])
    test_labels = np.array([-1, -1, 1, -1, -1])
    confusion = np.array([[2., 1.], [2., 0.]])
    accuracy = 0.6
    majority = np.array([-1, 1])
    ins = "\n X:" + str(features) + ",\nlabels:" + str(labels)
    topic = "Testing KNN(3) "

    def test_majority():
        outs = majority
        comment = topic + "majority_vote" + ins + "\n expected output: " + str(outs)
        knn = knnclass(3).fit(features, labels)
        obtained = knn.majority_vote(np.array([[1, 2], [3, 4]]), np.array([[0, 0], [0, 0]]))
        comment = comment + "\n obtained: " + str(obtained)
        if np.allclose(obtained, outs, atol=1e-5):
            return True, comment
        return False, comment

    def test_predict():
        outs = test_labels
        comment = topic + "predict" + ins + "\n expected output: " + str(outs)
        knn = knnclass(3).fit(features, labels)
        obtained = knn.predict(test_points)
        comment = comment + "\n obtained: " + str(obtained)
        if np.allclose(obtained, outs, atol=1e-5):
            return True, comment
        return False, comment

    def test_confusion():
        outs = confusion
        comment = topic + "confusion" + ins + "\n expected output: " + str(outs)
        knn = knnclass(3).fit(features, labels)
        obtained = knn.confusion_matrix(test_points, np.array([1, -1, -1, 1, -1]))
        comment = comment + "\n obtained: " + str(obtained)
        if np.allclose(obtained, outs, atol=1e-5):
            return True, comment
        return False, comment

    def test_accuracy():
        outs = accuracy
        comment = topic + "accuracy" + ins + "\n expected output: " + str(outs)
        knn = knnclass(3).fit(features, labels)
        obtained = knn.accuracy(test_points, np.array([1, 1, 1, -1, -1]))
        comment = comment + "\n obtained: " + str(obtained)
        if np.allclose(obtained, outs, atol=1e-5):
            return True, comment
        return False, comment

    tester.add_test("1.1", test_majority)
    tester.add_test("1.2", test_predict)
    tester.add_test("1.3", test_confusion)
    tester.add_test("1.4", test_accuracy)
    tester.run()




def test_threshold(threshold_func):
    features = np.array([
        [39, 38000, 0, 1],
        [48, 49000, 0, 0],
        [57, 92000, 0, 1],
        [38, 41000, 0, 1],
    ])
    labels = np.array([1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1])
    expected = np.array([[0, 0, 0, 1], [1, 0, 0, 0], [1, 1, 0, 1], [0, 0, 0, 1]])

    tester = Tester()
    ins = "\n X:" + str(features)
    topic = "Testing threshold_features(features, 40, 50000) "

    def test_thresh():
        outs = expected
        comment = topic + ins + "\n expected output: " + str(outs)
        obtained = threshold_func(features, 40, 50000)
        comment = comment + "\n obtained: " + str(obtained)
        if np.allclose(obtained, outs, atol=1e-5):
            return True, comment
        return False, comment

    tester.add_test("3.1", test_thresh)
    tester.run()


def test_NB(nb_class):
    features = np.array([
        [0, 1, 0, 1],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 1, 1]
    ])
    labels = np.array([1, 0, 1, 0, 0])
    classes_log = np.array([-0.51082562, -0.91629073])
    features_log = [np.array([[-0.40546511, -1.09861229],
                              [-0.69314718, -0.69314718]]),
                    np.array([[-1.09861229, -0.40546511],
                              [-0.69314718, -0.69314718]]),
                    np.array([[-0.40546511, -1.09861229],
                              [-0.69314718, -0.69314718]]),
                    np.array([[-1.09861229, -0.40546511],
                              [-0.69314718, -0.69314718]])]
    jll = np.array([[-1.62186043, -2.77258872],
                    [-2.31500761, -2.77258872],
                    [-4.39444915, -2.77258872],
                    [-2.31500761, -2.77258872],
                    [-3.00815479, -2.77258872]])
    predict = np.array([0, 0, 1, 0, 0])
    tester = Tester()
    ins = "\n X:" + str(features)
    topic = "Testing NaiveBayes "

    def test_classes():
        outs = classes_log
        comment = topic + "classes_log_prob" + ins + "\n expected output: " + str(outs)
        nb = nb_class().fit(features, labels)
        obtained = nb.classes_log_probability
        comment = comment + "\n obtained: " + str(obtained)
        if np.allclose(obtained, outs, atol=1e-5):
            return True, comment
        return False, comment

    def test_features():
        outs = features_log
        comment = topic + "features_log_likelhd" + ins + "\n expected output: " + str(outs)
        nb = nb_class().fit(features, labels)
        obtained = nb.features_log_likelihood
        comment = comment + "\n obtained: " + str(obtained)
        if np.allclose(obtained, outs, atol=1e-5):
            return True, comment
        return False, comment

    def test_jll():
        outs = jll
        comment = topic + "joint_log_likelhd(features)" + ins + "\n expected output: " + str(outs)
        nb = nb_class().fit(features, labels)
        obtained = nb.joint_log_likelihood(features)
        comment = comment + "\n obtained: " + str(obtained)
        if np.allclose(obtained, outs, atol=1e-5):
            return True, comment
        return False, comment
    def test_predict():
        outs = predict
        comment = topic + "predict(features)" + ins + "\n expected output: " + str(outs)
        nb = nb_class().fit(features, labels)
        obtained = nb.predict(features)
        comment = comment + "\n obtained: " + str(obtained)
        if np.allclose(obtained, outs, atol=1e-5):
            return True, comment
        return False, comment
    tester.add_test("3.3", test_classes)
    tester.add_test("3.4", test_features)
    tester.add_test("3.5", test_jll)
    tester.add_test("3.6", test_predict)
    tester.run()
