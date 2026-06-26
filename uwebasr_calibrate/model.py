import os
import logging
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

logger = logging.getLogger(__name__)

class CalibratedPredictor:
    """
    Combines HistGradientBoostingRegressor and affine calibration.
    """
    def __init__(self, model, a, b):
        self.model = model
        self.a = a
        self.b = b
        
    def predict(self, X):
        raw_pred = self.model.predict(X)
        calibrated_pred = self.a + self.b * raw_pred
        return np.clip(calibrated_pred, 0.0, 1.0)

def train_calibration_model(train_samples, seed=13, loss_metric="mae"):
    """
    Trains the calibrated accuracy predictor using HistGradientBoostingRegressor and Grid Search.
    """
    if loss_metric not in ["mae", "mse"]:
        raise ValueError(f"Unsupported loss_metric: {loss_metric}. Choose 'mae' or 'mse'.")
        
    # Prepare matrices
    X = np.array([s["features"] for s in train_samples])
    y = np.array([s["accuracy"] for s in train_samples])
    
    # Split train into fit and validation subsets (80/20)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    
    # Grid search grid
    learning_rates = [0.02, 0.05]
    max_leaf_nodes_list = [15, 31]
    min_samples_leaf_list = [40, 200]
    l2_regs = [0.0, 0.1]
    
    best_val_score = float("inf")
    best_params = None
    
    logger.info(f"Starting hyperparameter grid search optimizing for {loss_metric.upper()}...")
    
    hgbr_loss = "absolute_error" if loss_metric == "mae" else "squared_error"
    
    for lr in learning_rates:
        for max_leaf in max_leaf_nodes_list:
            for min_samples in min_samples_leaf_list:
                for l2 in l2_regs:
                    params = {
                        "learning_rate": lr,
                        "max_leaf_nodes": max_leaf,
                        "min_samples_leaf": min_samples,
                        "l2_regularization": l2,
                        "max_iter": 700,
                        "loss": hgbr_loss,
                        "random_state": seed
                    }
                    
                    model = HistGradientBoostingRegressor(**params)
                    model.fit(X_fit, y_fit)
                    
                    # Evaluate on validation
                    val_pred = model.predict(X_val)
                    if loss_metric == "mae":
                        val_score = np.mean(np.abs(y_val - val_pred))
                    else:
                        val_score = np.mean((y_val - val_pred) ** 2)
                        
                    logger.debug(f"Params: lr={lr}, max_leaf={max_leaf}, min_samples={min_samples}, l2={l2} -> Val {loss_metric.upper()}: {val_score:.5f}")
                    
                    if val_score < best_val_score:
                        best_val_score = val_score
                        best_params = params
                        
    logger.info(f"Best params: {best_params} with Val {loss_metric.upper()}: {best_val_score:.5f}")
    
    # Refit on the complete train partition
    final_model = HistGradientBoostingRegressor(**best_params)
    final_model.fit(X, y)
    
    # Get final model's predictions on complete train partition for affine calibration
    train_raw_preds = final_model.predict(X)
    
    # Fit affine calibration (Linear Regression)
    lr_calib = LinearRegression()
    lr_calib.fit(train_raw_preds.reshape(-1, 1), y)
    
    a = float(lr_calib.intercept_)
    b = float(lr_calib.coef_[0])
    
    logger.info(f"Fitted affine calibration: Acc_hat = clip({a:.5f} + {b:.5f} * HGBR_pred, 0, 1)")
    
    # Create final calibrated predictor
    predictor = CalibratedPredictor(final_model, a, b)
    
    # Get validation predictions with the final calibrated model
    val_pred_raw = final_model.predict(X_val)
    val_pred_calib = np.clip(a + b * val_pred_raw, 0.0, 1.0)
    
    return predictor, best_params, best_val_score, val_pred_calib, y_val
