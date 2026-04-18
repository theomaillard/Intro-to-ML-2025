import time

import numpy as np
import itertools

from matplotlib import pyplot as plt

from src.utils import macrof1_fn



class KMeans(object):
    """
    kNN classifier object.
    """

    NUMBER_INITIALIZATIONS = 10

    def __init__(self, max_iters=500):
        """
        Call set_arguments function of this class.
        """
        self.max_iters = max_iters
        self.centroids = None
        self.best_permutation = None


    def _init_centers(self, data, number_of_centers):
        number_of_rows = data.shape[0]
        random_indices = np.random.permutation(number_of_rows)
        centers_indices = random_indices[:number_of_centers]
        centers = data[centers_indices]
        return centers

    def _compute_distances(self, data, centers):
        """
        Computer a matrix of distances from a point to each cluster, for each point.
        Returns a matrix of shape (N, K) with N = number of points and K = number of clusters.

        Arguments:
            data: array of shape (N, D) where N is the number of data points, D is the number of features (:=pixels).
            centers: array of shape (K, D), centers of the K clusters.
        Returns:
            distances: array of shape (N, K) with the distances between the N points and the K clusters.
        """

        N = data.shape[0]
        K = centers.shape[0]

        ''' -------   # More efficient version below -------
        distance_matrix = np.zeros((N, K))
        for i in range(K):
            center = centers[i]
            distance_matrix[:, i] = np.sqrt(((data - center) ** 2).sum(axis=1))
        '''

        distance_matrix = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)

        return distance_matrix

    def _find_closest_cluster(self, distances):
        """
        Assign datapoints to the closest clusters.

        Arguments:
            distances: array of shape (N, K), the distance of each data point to each cluster center.
        Returns:
            cluster_assignments: array of shape (N,), cluster assignment of each datapoint, which are an integer between 0 and K-1.
        """
        return np.argmin(distances, axis=1)   # For each row, it returns the column with the lowest value


    def _compute_centers(self, data, assigned_clusters, k):
        """
            Compute the center of each cluster based on the assigned points.

            Arguments:
                data: data array of shape (N,D), where N is the number of samples, D is number of features
                cluster_assignments: the assigned cluster of each data sample as returned by find_closest_cluster(), shape is (N,)
                K: the number of clusters
            Returns:
                centers: the new centers of each cluster, shape is (K,D) where K is the number of clusters, D the number of features
            """

        N, D = data.shape   # Equivalent to doing N = data.shape[0]  D = data.shape[1]
        centers = np.zeros((k, D))

        for i in range(k):
            condition = assigned_clusters == i

            # Handles the case where a cluster is empty
            if np.any(condition):
                centers[i] = data[condition].mean(axis=0)
            else:
                centers[i] = data[np.random.choice(N)]

        return centers

    def _assign_labels_to_centers(self, centers, cluster_assignments, true_labels):
        """
        Use voting to attribute a label to each cluster center.

        Arguments:
            centers: array of shape (K, D), cluster centers
            cluster_assignments: array of shape (N,), cluster assignment for each data point.
            true_labels: array of shape (N,), true labels of data
        Returns:
            cluster_center_label: array of shape (K,), the labels of the cluster centers
        """
        cluster_center_label = np.zeros(centers.shape[0])
        for i in range(len(centers)):
            labels = true_labels[cluster_assignments == i]
            if labels.size == 0:
                cluster_center_label[i] = np.random.choice(true_labels)
            else:
                values_unique, counts = np.unique(labels, return_counts=True)
                cluster_center_label[i] = values_unique[np.argmax(counts)]

        return cluster_center_label


    def _predict_with_centers(self, data, centers, cluster_center_label):
        """
        Predict the label for data, given the cluster center and their labels.
        To do this, it first assigns points in data to their closest cluster, then use the label
        of that cluster as prediction.

        Arguments:
            data: array of shape (N, D)
            centers: array of shape (K, D), cluster centers
            cluster_center_label: array of shape (K,), the labels of the cluster centers
        Returns:
            new_labels: array of shape (N,), the labels assigned to each data point after clustering, via k-means.
        """
        # Compute cluster assignments
        distances = self._compute_distances(data, centers)
        cluster_assignments = self._find_closest_cluster(distances)

        # Convert cluster index to label
        new_labels = cluster_center_label[cluster_assignments]
        return new_labels


    def _compute_accuracy(self, predicted_labels, true_labels):
        return np.mean(predicted_labels == true_labels)

    def fit(self, training_data, training_labels, chosen_k=0):
        """
        Trains the model, returns predicted labels for training data.
        Hint:
            (1) Since Kmeans is unsupervised clustering, we don't need the labels for training. But you may want to use it to determine the number of clusters.
            (2) Kmeans is sensitive to initialization. You can try multiple random initializations when using this classifier.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): labels of shape (N,).
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """

        k = chosen_k if chosen_k > 0 else len(np.unique(training_labels))
        self.centroids = None
        self.best_permutation = None

        best_accuracy = -np.inf
        best_centers = None
        best_center_labels = None
        best_pred_labels = None

        # Testing different initializations
        for _ in range(self.NUMBER_INITIALIZATIONS):
            self.centroids = self._init_centers(training_data, k)

            # Iterating until the algorithm converges
            for _ in range(self.max_iters):
                distance_matrix = self._compute_distances(training_data, self.centroids)
                assigned_clusters = self._find_closest_cluster(distance_matrix)
                new_centers = self._compute_centers(training_data, assigned_clusters, k)

                if np.allclose(self.centroids, new_centers, atol=1e-5):
                    break  # The algorithm converged

                self.centroids = new_centers

            center_labels = self._assign_labels_to_centers(self.centroids, assigned_clusters, training_labels)
            pred_labels = self._predict_with_centers(training_data, self.centroids, center_labels)

            # f1 score is a better metric than accuracy
            accuracy = macrof1_fn(pred_labels, training_labels)
            #accuracy = self._compute_accuracy(pred_labels, training_labels)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_centers = self.centroids
                best_center_labels = center_labels
                best_pred_labels = pred_labels

        self.centroids = best_centers
        self.best_permutation = best_center_labels
        return best_pred_labels


    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            test_labels (np.array): labels of shape (N,)
        """

        test_labels = self._predict_with_centers(test_data, self.centroids, self.best_permutation)
        return test_labels

    def optimize_parameters(self, test_data, test_labels, training_data, training_labels, kmax=0, plot=False):
        """
            Optimizes parameter k of the model by testing all possibilities from k=1 to k=kmax or k=n.
        """

        start_time = time.time()

        if kmax == 0:
            kmax = training_data.shape[0]

        # Find the k that yields the highest F1 score
        accuracy = np.zeros(kmax)
        accuracy_train = np.zeros(kmax)
        macrof1 = np.zeros(kmax)
        macrof1_train = np.zeros(kmax)
        for i in range(1, kmax):
            current_k = i
            self.fit(training_data, training_labels, current_k)
            predicted_test_labels = self.predict(test_data)
            predicted_train_labels = self.predict(training_data)
            accuracy[i] = self._compute_accuracy(predicted_test_labels, test_labels)
            accuracy_train[i] = self._compute_accuracy(predicted_train_labels, training_labels)
            macrof1[i] = macrof1_fn(predicted_test_labels, test_labels)
            macrof1_train[i] = macrof1_fn(predicted_train_labels, training_labels)

        # Display data about the experiment
        best_k = np.argmax(macrof1)
        print(f"Best model for k = {best_k}: with f1 score = {macrof1[best_k]:.3f} and accuracy = {100*accuracy[best_k]:.1f}%")
        end_time = time.time()
        elapsed_seconds = (end_time - start_time)
        print(f"It took {elapsed_seconds:.2f} seconds to find the best model.")

        # Plotting the results
        plt.plot(macrof1, label="Macro F1 Score")
        plt.plot(accuracy, label="Accuracy")
        plt.plot(macrof1_train, label="Macro F1 Score in training", linestyle="--")
        plt.plot(accuracy_train, label="Accuracy in training", linestyle='--')
        plt.xlim([1, kmax])
        plt.xlabel("k - Number of clusters")
        plt.ylabel("Metric of algorithm")
        plt.legend()
        plt.title("Hyperparameter (zoomed)")
        plt.show()
        plt.close()
