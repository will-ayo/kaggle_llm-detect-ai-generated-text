import numpy as np

from functools import partial
from scipy.optimize import fmin

from sklearn import metrics


class OptimizeAUC:
    """
    Class for optimizing AUC by finding the best linear combination of model predictions.

    Attributes:
        coef_ (ndarray): The best coefficients found for the ensemble.
    """

    def __init__(self):
        self.coef_ = None

    def _negative_auc(self, coef, X, y):
        """
        Compute negative AUC for a given set of ensemble coefficients.

        Args:
            coef (ndarray): Coefficients for each model (same length as # models).
            X (ndarray): Predictions from multiple models (shape: [n_samples, n_models]).
            y (ndarray): True labels (shape: [n_samples]).

        Returns:
            float: Negative AUC score.
        """
        # Weighted predictions
        predictions = np.sum(X * coef, axis=1)
        auc_score = metrics.roc_auc_score(y, predictions)
        return -auc_score

    def fit(self, X, y):
        """
        Fit the ensemble coefficients to maximize AUC (minimize negative AUC).

        Args:
            X (ndarray): Predictions from multiple models (shape: [n_samples, n_models]).
            y (ndarray): True labels (shape: [n_samples]).
        """
        from numpy.random import dirichlet
        loss_partial = partial(self._negative_auc, X=X, y=y)

        # Initialize coefficients from a Dirichlet distribution (to sum to 1).
        initial_coef = dirichlet(np.ones(X.shape[1]), size=1)[0]

        # Minimize negative AUC
        self.coef_ = fmin(loss_partial, initial_coef, disp=True)

    def predict(self, X):
        """
        Generate ensemble predictions with the learned coefficients.

        Args:
            X (ndarray): Predictions from multiple models (shape: [n_samples, n_models]).

        Returns:
            ndarray: Final predictions (shape: [n_samples]).
        """
        return np.sum(X * self.coef_, axis=1)