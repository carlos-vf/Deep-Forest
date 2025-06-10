"""
Implementation of the estimator wrapper to support customized base estimators.
"""


__all__ = ["KFoldWrapper"]

import copy
import numpy as np
from sklearn.model_selection import KFold

from .. import _utils


class KFoldWrapper(object):
    """
    A general wrapper for base estimators without the characteristic of
    out-of-bag (OOB) estimation.
    MODIFIED to support uncertainty propagation.
    """

    def __init__(
        self,
        estimator,
        n_splits,
        n_outputs,
        random_state=None,
        verbose=1,
        is_classifier=True,
    ):

        # Parameters were already validated by upstream methods
        self.dummy_estimator_ = estimator
        self.n_splits = n_splits
        self.n_outputs = n_outputs
        self.random_state = random_state
        self.verbose = verbose
        self.is_classifier = is_classifier
        # Internal container
        self.estimators_ = []

    @property
    def estimator_(self):
        """Return the list of internal estimators."""
        return self.estimators_

    # Accepts dX and returns both mean and dX OOB predictions
    def fit_transform(self, X, y, dX=None, py=None, sample_weight=None):
        n_samples, _ = X.shape
        splitter = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        # OOB arrays for mean and standard deviation
        self.oob_decision_function_ = np.zeros((n_samples, self.n_outputs))
        self.oob_decision_function_dX_ = np.zeros((n_samples, self.n_outputs))

        for k, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
            estimator = copy.deepcopy(self.dummy_estimator_)

            if self.verbose > 1:
                msg = "{} - - Fitting the base estimator with fold = {}"
                print(msg.format(_utils.ctime(), k))
            
            # Prepare data for this fold
            X_train, y_train = X[train_idx], y[train_idx]
            X_val = X[val_idx]
            X_train_dX = dX[train_idx] if dX is not None else None
            X_val_dX = dX[val_idx] if dX is not None else None
            py_train = py[train_idx] if py is not None else None

            # Fit on training samples
            # Assumes your custom estimator's `fit` can handle dX if needed.
            fit_args = {"dX": X_train_dX, "py": py_train}
            if sample_weight is None:
                estimator.fit(X_train, y_train, **fit_args)
            else:
                estimator.fit(
                    X_train, y_train, sample_weight=sample_weight[train_idx], **fit_args
                )

            # Predict on hold-out samples
            # both the mean prediction and its standard deviation.
            if self.is_classifier:
                if not hasattr(estimator, "predict_proba_with_dX"):
                    msg = ("Custom estimator must have a `predict_proba_with_dX` "
                           "method to be used for uncertainty estimation.")
                    raise AttributeError(msg)
                
                # Get both mean and dX from the custom estimator
                mean_pred, dX_pred = estimator.predict_proba_with_dX(X_val, dX=X_val_dX)
                self.oob_decision_function_[val_idx] += mean_pred
                self.oob_decision_function_dX_[val_idx] += dX_pred
            else: # Regression
                if not hasattr(estimator, "predict_with_dX"):
                     msg = ("Custom regressor must have a `predict_with_dX` "
                           "method to be used for uncertainty estimation.")
                     raise AttributeError(msg)
                
                mean_pred, dX_pred = estimator.predict_with_dX(X_val, dX=X_val_dX)
                
                # Reshape for univariate regression
                if self.n_outputs == 1:
                    if len(mean_pred.shape) == 1:
                        mean_pred = np.expand_dims(mean_pred, 1)
                    if len(dX_pred.shape) == 1:
                        dX_pred = np.expand_dims(dX, 1)
                        
                self.oob_decision_function_[val_idx] += mean_pred
                self.oob_decision_function_dX_[val_idx] += dX_pred

            # Store the estimator
            self.estimators_.append(estimator)
        
        return self.oob_decision_function_, self.oob_decision_function_dX_


    def predict(self, X, dX=None):
        n_samples, _ = X.shape
        
        all_predictions = []

        for estimator in self.estimators_:
            # During inference, we rely on the standard predict_proba/predict
            # and calculate the ensemble uncertainty ourselves.
            predict_args = {"dX": dX} if dX is not None else {}

            if self.is_classifier:
                pred = estimator.predict_proba(X, **predict_args)
                all_predictions.append(pred)
            else: # Regression
                pred = estimator.predict(X, **predict_args)
                if self.n_outputs == 1:
                    pred = pred.reshape(n_samples, -1)
                all_predictions.append(pred)

        all_predictions = np.array(all_predictions) # Shape: (n_splits, n_samples, n_outputs)
        
        mean_pred = np.mean(all_predictions, axis=0)
        dX_pred = np.std(all_predictions, axis=0)

        # ADD THIS BLOCK FOR VERIFICATION
        # ===================================================================
        #print("\n--- [Verification Step 4: KFoldWrapper Inference] ---")
        #print(f"Shape of all predictions collected: {all_predictions.shape}")
        #print("Sample of final mean prediction:\n", mean_pred[:2, :])
        #print("Sample of final dX dev prediction:\n", dX_pred[:2, :])
        # ===================================================================

        return mean_pred, dX_pred