import numpy as np
import matplotlib.pyplot as plt

# Generally utilizes
##################
def label_to_onehot(labels, C=None):
    """
    Transform the labels into one-hot representations.

    Arguments:
        labels (np.array): labels as class indices, of shape (N,)
        C (int): total number of classes. Optional, if not given
                 it will be inferred from labels.
    Returns:
        one_hot_labels (np.array): one-hot encoding of the labels, of shape (N,C)
    """
    N = labels.shape[0]
    if C is None:
        C = get_n_classes(labels)
    one_hot_labels = np.zeros([N, C])
    one_hot_labels[np.arange(N), labels.astype(int)] = 1
    return one_hot_labels


def onehot_to_label(onehot):
    """
    Transform the labels from one-hot to class index.

    Arguments:
        onehot (np.array): one-hot encoding of the labels, of shape (N,C)
    Returns:
        (np.array): labels as class indices, of shape (N,)
    """
    return np.argmax(onehot, axis=1)


def append_bias_term(data):
    """
    Append to the data a bias term equal to 1.

    Arguments:
        data (np.array): of shape (N,D)
    Returns:
        (np.array): shape (N,D+1)
    """
    N = data.shape[0]
    data = np.concatenate([np.ones([N, 1]), data], axis=1)
    return data

def normalize(data):
    """
    Return the normalized data, computing means and stds through numpy.

    Arguments:
        data (np.array): of shape (N,D)
    Returns:
        normalized_data (np.array): shape (N,D)
    """
    std = np.std(data)
    mean = np.mean(data)
    return normalize_fn(data, std, mean)

def normalize_fn(data, means, stds):
    """
    Return the normalized data, based on precomputed means and stds.
    
    Arguments:
        data (np.array): of shape (N,D)
        means (np.array): of shape (1,D)
        stds (np.array): of shape (1,D)
    Returns:
        (np.array): shape (N,D)
    """
    # return the normalized features
    return (data - means) / stds


def get_n_classes(labels):
    """
    Return the number of classes present in the data labels.
    
    This is approximated by taking the maximum label + 1 (as we count from 0).
    """
    return int(np.max(labels) + 1)


# Metrics
#########
def accuracy_fn(pred_labels, gt_labels):
    """
    Return the accuracy of the predicted labels.
    """
    return np.mean(pred_labels == gt_labels)


def macrof1_fn(pred_labels, gt_labels):
    """
    Return the macro F1-score.

    Arguments:
        pred_labels (np.array):
        gt_labels (np.array):
    Returns:

    """
    class_ids = np.unique(gt_labels)
    macrof1 = 0
    for val in class_ids:
        predpos = (pred_labels == val)
        gtpos = (gt_labels == val)

        tp = sum(predpos * gtpos)
        fp = sum(predpos * ~gtpos)
        fn = sum(~predpos * gtpos)
        if tp == 0:
            continue
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)

        macrof1 += 2 * (precision * recall) / (precision + recall)

    return macrof1 / len(class_ids)


def mse_fn(pred, gt):
    """
    Mean Squared Error
    Arguments:
        pred: NxD prediction matrix
        gt: NxD groundtruth values for each predictions
    Returns:
        returns the computed loss
    """
    loss = (pred - gt) ** 2
    loss = np.mean(loss)
    return loss

def plot_confusion_matrix(y_true, y_pred):
    """
    Function resembling the scikit-learn function for plotting a confusion matrix with the inputs
    y_true (true labels), y_pred (predicted labels).
    """
    classes = np.unique(y_true)
    matrix = np.zeros((len(classes), len(classes)))

    for i in range(len(classes)):
        for j in range(len(classes)):
            matrix[i, j] = np.sum((y_true == classes[i]) & (y_pred == classes[j]))

    fig, ax = plt.subplots()
    ax.matshow(matrix, cmap='Blues')

    ax.set_xticks(range(len(classes)), labels=[f'Class {i}' for i in range(len(classes))])
    ax.set_yticks(range(len(classes)), labels=[f'Class {i}' for i in range(len(classes))])
    ax.xaxis.set_ticks_position('bottom')

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(matrix[i, j]), ha='center', va='center', color='orange')

    plt.title('Confusion Matrix for Test Data')
    plt.show()