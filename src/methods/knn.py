import numpy as np
import matplotlib.pyplot as plt
from ..utils import get_n_classes, accuracy_fn, macrof1_fn

class KNN(object):
    """
        kNN classifier object. by Dino Cheng

        kNN mostly adapted from the homework session 3 of the CS-233 class
        of EPF Lausanne in spring semester 2025,
        where a similar algorithm has been implemented before.

        The parallelization/broadcasting might not have been acccomplished
        to the fullest (theoretically) possible theoretical limit.
        While numpy operations have been used where possible, broadcasting has
        only been implemented on the most-outer loop (the predict function calling
        predict_single for each single test sample). However, for our case, further broadcasting would not bring any big improvements for the already short runtime.
    """

    def __init__(self, k=1, task_kind = "classification"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """

        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##

        self.D, self.C = training_data.shape[1], get_n_classes(training_labels)
        self.training_data = training_data
        self.training_data_size = self.training_data.shape[0]
        self.training_labels = training_labels.astype(int) # no support for float values, as not yet required
        return training_labels # what is there to return properly? all training data is already fitted by definition

    @staticmethod
    def euclidean_dist(example, training_examples):
        """Compute the Euclidean distance between a single example
        vector and all training_examples.

        Inputs:
            example: shape (D,)
            training_examples: shape (NxD)
        Outputs:
            euclidean distances: shape (N,)
        """
        # WRITE YOUR CODE HERE
        return np.linalg.norm(example - training_examples, 2, axis=1)

    @staticmethod
    def find_k_nearest_neighbors(k, distances):
        """ Find the indices of the k smallest distances from a list of distances.
            Tip: use np.argsort()

        Inputs:
            k: integer
            distances: shape (N,)
        Outputs:
            indices of the k nearest neighbors: shape (k,)
        """
        # WRITE YOUR CODE HERE
        indices = np.argsort(distances)[:k]
        return indices

    @staticmethod
    def predict_label(neighbor_labels):
        """Return the most frequent label in the neighbors'.

        Inputs:
            neighbor_labels: shape (N,)
        Outputs:
            most frequent label
        """
        # WRITE YOUR CODE HERE
        return np.argmax(np.bincount(neighbor_labels))

    def predict_single(self, test_data):
        """
        Runs kNN prediction on a single test data and compares it against its nearest neighbors
        from self.training_labels.
        """
        distances = KNN.euclidean_dist(test_data, self.training_data)

        neighbors_indices = KNN.find_k_nearest_neighbors(self.k, distances)

        neighbor_labels = self.training_labels[neighbors_indices]

        label = KNN.predict_label(neighbor_labels)
        return label

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Without parallelization, the code might look somewhat like this:
            test_labels = np.zeros(test_data.shape[0])
            for i, x in enumerate(test_data):
                test_labels[i] = self.predict_single(x)
            return test_labels

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##
        return np.apply_along_axis(
            self.predict_single,
            axis=1,
            arr=test_data
        )

    def optimize_parameters(self, test_data, ytest, train_data, train_labels, kmax=0, plot=True, optimize_f1=True):
        """
        Optimizes parameter k of the model by testing all possibilities from k=1 to k=kmax or k=n.
        """
        if kmax == 0:
            kmax = self.training_data.shape[0]
        self.k = 1

        # find highest accuracy
        accuracy = np.zeros(kmax)
        accuracy_train = np.zeros(kmax)
        macrof1 = np.zeros(kmax)
        macrof1_train = np.zeros(kmax)
        for i in np.arange(1, kmax):
            self.k = i
            test_labels = self.predict(test_data)

            train_labels = self.predict(self.training_data)
            accuracy[i] = accuracy_fn(test_labels, ytest)
            accuracy_train[i] = accuracy_fn(train_labels, self.training_labels)
            macrof1[i] = macrof1_fn(test_labels, ytest)
            macrof1_train[i] = macrof1_fn(train_labels, self.training_labels)

        best_f1 = np.max(macrof1)
        best_accuracy = np.max(accuracy)
        if optimize_f1:
            self.k = np.argmax(macrof1)
        else:
            self.k = np.argmax(accuracy)
        print(f"Best model for k = {self.k}: with f1 score = {macrof1[self.k]} and accuracy = {accuracy[self.k]}")
        if plot:
            plt.plot(macrof1, label="Macro F1 Score")
            plt.plot(accuracy, label="Accuracy")
            plt.plot(macrof1_train, label="Macro F1 Score in training", linestyle="--")
            plt.plot(accuracy_train, label="Accuracy in training", linestyle='--')
            plt.xlim([1, max(24,int(self.training_data_size/4))]) # k > self.training_data_size / 2 is rarely useful
            plt.xlabel("k - Number of Neighbors")
            plt.ylabel("Metric of algorithm")
            plt.legend()
            plt.title("Hyperparameter (zoomed)")
            plt.show()
            plt.close()
