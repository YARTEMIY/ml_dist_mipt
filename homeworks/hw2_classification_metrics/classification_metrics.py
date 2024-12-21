import numpy as np


class ClassificationMetrics:
    @staticmethod
    def accuracy(labels, preds):
        """
        labels : numpy array of shape (`n_observations`)
        predictions : numpy array of shape (`n_observations`)

        Return : float
            single number with accuracy score
        
        Comment: Both labels and predictions contain integers 0 or 1
        """

        # YOUR CODE HERE
        sum_equals = np.sum(labels == preds)
        return sum_equals / len(preds)

    @staticmethod
    def precision(labels, preds):
        """
        labels : numpy array of shape (`n_observations`)
        predictions : numpy array of shape (`n_observations`)

        Return : float
            single number with precision score
        
        Comment: Both labels and predictions contain integers 0 or 1
        """

        # YOUR CODE HERE
        true_positive = np.sum((labels == 1) & (preds == 1))
        false_positive = np.sum((labels == 0) & (preds == 1))
        return true_positive / (true_positive + false_positive)

    @staticmethod
    def recall(labels, preds):
        """
        labels : numpy array of shape (`n_observations`)
        predictions : numpy array of shape (`n_observations`)

        Return : float
            single number with recall score
        
        Comment: Both labels and predictions contain integers 0 or 1
        """
        
        # YOUR CODE HERE
        true_positive = np.sum((labels == 1) & (preds == 1))
        false_negative = np.sum((labels == 1) & (preds == 0))
        return true_positive / (true_positive + false_negative)

    @staticmethod
    def f1(labels, preds):
        """
        labels : numpy array of shape (`n_observations`)
        predictions : numpy array of shape (`n_observations`)

        Return : float
            single number with f1 score
        
        Comment: Both labels and predictions contain integers 0 or 1
        """

        # YOUR CODE HERE
        prec = ClassificationMetrics.precision(labels, preds)
        rec = ClassificationMetrics.recall(labels, preds)
        return 2 * prec * rec / (prec + rec)

    @staticmethod
    def f_beta(labels, preds, beta=1):
        """
        labels : numpy array of shape (`n_observations`)
        predictions : numpy array of shape (`n_observations`)
        beta : float

        Return : float
            single number with f_beta score
        
        Comment: Both labels and predictions contain integers 0 or 1
        """

        # YOUR CODE HERE
        prec = ClassificationMetrics.precision(labels, preds)
        rec = ClassificationMetrics.recall(labels, preds)
        return (1 + beta**2) * prec * rec / (beta**2 * prec + rec)
    