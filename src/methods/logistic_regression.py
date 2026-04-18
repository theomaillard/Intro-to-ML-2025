import numpy as np

from src.utils import get_n_classes, accuracy_fn, macrof1_fn, normalize, onehot_to_label, label_to_onehot, plot_confusion_matrix


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.weights = None
        self.lr = lr
        self.max_iters = max_iters

    @staticmethod
    def softmax(data, W):
        """
        Softmax function for multi-class logistic regression.

        Args:
            data (array): Input data of shape (N, D)
            W (array): Weights of shape (D, C) where C is the number of classes
        Returns:
            array of shape (N, C): Probability array where each value is in the
                range [0, 1] and each row sums to 1.
                The row i corresponds to the prediction of the ith data sample, and
                the column j to the jth class. So element [i, j] is P(y_i=k | x_i, W)
        """
        x_w = np.dot(data, W)
        x_w_shift = x_w - np.max(x_w, axis=1, keepdims=True) # to avoid overflow error

        return np.exp(x_w_shift) / np.sum(np.exp(x_w_shift), axis=1).reshape(-1, 1)

    @staticmethod
    def loss_function(data, labels, w):
        """
        Loss function for multi class logistic regression, i.e., multi-class entropy.

        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            w (array): Weights of shape (D, C)
        Returns:
            float: Loss value
        """
        y_pred = LogisticRegression.softmax(data, w)
        loss = -np.sum(labels * np.log(y_pred))

        return float(loss)

    @staticmethod
    def gradient(data, labels, W):
        """
        Compute the gradient of the entropy for multi-class logistic regression.

        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            W (array): Weights of shape (D, C)
        Returns:
            grad (np.array): Gradients of shape (D, C)
        """
        y_pred = LogisticRegression.softmax(data, W)
        difference = y_pred - labels
        gradient = np.matmul(data.T, difference)

        return gradient

    @staticmethod
    def logistic_regression_predict_multi(data, W):
        """
        Prediction the label of data for multi-class logistic regression.

        Args:
            data (array): Dataset of shape (N, D).
            W (array): Weights of multi-class logistic regression model of shape (D, C)
        Returns:
            array of shape (N,): Label predictions of data.
        """
        pred = LogisticRegression.softmax(data, W)

        return onehot_to_label(pred)

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        D = training_data.shape[1]  # number of features
        C = get_n_classes(training_labels)
        one_hot_labels = label_to_onehot(training_labels, C)
        self.xtrain = training_data
        self.ytrain = training_labels

        self.weights = np.random.normal(0, 0.1, (D, C))

        for it in range(self.max_iters):
            gradient = LogisticRegression.gradient(training_data, one_hot_labels, self.weights)
            self.weights = self.weights - self.lr * gradient
            predictions = LogisticRegression.logistic_regression_predict_multi(training_data, self.weights)
            if accuracy_fn(predictions, onehot_to_label(one_hot_labels)) == 100:
                break

        return LogisticRegression.logistic_regression_predict_multi(training_data, self.weights)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        return LogisticRegression.logistic_regression_predict_multi(test_data, self.weights)

    def optimize_parameters(self, xtest, ytest, xtrain, ytrain, plot=False):
        """
        Grid search to find the best learning rate and max iterations.
        """
        learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.5]
        max_iterations = [100, 1000, 5000, 10000, 50000, 100000]

        best_f1 = 0
        acc_associated_with_best_f1 = 0
        best_lr_f1 = 0
        best_iter_f1 = 0
        best_model_f1 = None

        best_acc = 0
        f1_associated_with_best_acc = 0
        best_lr_acc = 0
        best_iter_acc = 0
        best_model_acc = None

        for lr in learning_rates:
            for it in max_iterations:
                np.random.seed(100)
                model = LogisticRegression(lr=lr, max_iters=it)
                model.fit(self.xtrain, self.ytrain)
                preds_test = model.predict(xtest)

                acc = accuracy_fn(preds_test, ytest)
                macrof1 = macrof1_fn(preds_test, ytest)

                if macrof1 > best_f1:
                    best_f1 = macrof1
                    best_lr_f1 = lr
                    best_iter_f1 = it
                    acc_associated_with_best_f1 = acc
                    best_model_f1 = model

                if acc > best_acc:
                    best_acc = acc
                    best_lr_acc = lr
                    best_iter_acc = it
                    f1_associated_with_best_acc = macrof1
                    best_model_acc = model

        print(f"\nBest F1 hyperparameters:")
        print(f"lr: {best_lr_f1}, iterations: {best_iter_f1}")
        print(f"F1: {best_f1}")
        print(f"accuracy: {acc_associated_with_best_f1 * 100}")

        print(f"\nBest Accuracy hyperparameters:")
        print(f"lr: {best_lr_acc}, iterations: {best_iter_acc}")
        print(f"Accuracy: {best_acc * 100}")
        print(f"F1: {f1_associated_with_best_acc}")

        print("\n =================== Model with best F1 hyperparameters ================")
        model = best_model_f1
        preds_train = model.predict(self.xtrain)
        preds_test = model.predict(xtest)

        train_acc = accuracy_fn(preds_train, self.ytrain)
        train_f1 = macrof1_fn(preds_train, self.ytrain)
        print(f"Train set: accuracy = {train_acc * 100:.3f}% - F1-score = {train_f1:.6f}")

        test_acc = accuracy_fn(preds_test, ytest)
        test_f1 = macrof1_fn(preds_test, ytest)
        print(f"Test set: accuracy = {test_acc * 100:.3f}% - F1-score = {test_f1:.6f}")

        print("\n =================== Model with best accuracy hyperparameters ================")
        model = best_model_acc
        preds_train = model.predict(self.xtrain)
        preds_test = model.predict(xtest)

        train_acc = accuracy_fn(preds_train, self.ytrain)
        train_f1 = macrof1_fn(preds_train, self.ytrain)
        print(f"Train set: accuracy = {train_acc * 100:.3f}% - F1-score = {train_f1:.6f}")

        test_acc = accuracy_fn(preds_test, ytest)
        test_f1 = macrof1_fn(preds_test, ytest)
        print(f"Test set: accuracy = {test_acc * 100:.3f}% - F1-score = {test_f1:.6f}")
