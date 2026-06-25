import os
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

logger = logging.getLogger(__name__)

class CalibratedPredictor:
    """
    Combines feature standardization, HistGradientBoostingRegressor, and affine calibration.
    """
    def __init__(self, scaler, model, a, b):
        self.scaler = scaler
        self.model = model
        self.a = a
        self.b = b
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        raw_pred = self.model.predict(X_scaled)
        calibrated_pred = self.a + self.b * raw_pred
        return np.clip(calibrated_pred, 0.0, 1.0)

def train_calibration_model(train_samples, seed=13):
    """
    Trains the calibrated accuracy predictor using HistGradientBoostingRegressor and Grid Search.
    """
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
    
    best_val_mae = float("inf")
    best_params = None
    
    logger.info("Starting hyperparameter grid search...")
    
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
                        "loss": "absolute_error",
                        "random_state": seed
                    }
                    
                    # Fit preprocessing on fit subset only
                    scaler = StandardScaler()
                    X_fit_scaled = scaler.fit_transform(X_fit)
                    X_val_scaled = scaler.transform(X_val)
                    
                    model = HistGradientBoostingRegressor(**params)
                    model.fit(X_fit_scaled, y_fit)
                    
                    # Evaluate on validation
                    val_pred = model.predict(X_val_scaled)
                    mae = np.mean(np.abs(y_val - val_pred))
                    
                    logger.debug(f"Params: lr={lr}, max_leaf={max_leaf}, min_samples={min_samples}, l2={l2} -> Val MAE: {mae:.5f}")
                    
                    if mae < best_val_mae:
                        best_val_mae = mae
                        best_params = params
                        
    logger.info(f"Best params: {best_params} with Val MAE: {best_val_mae:.5f}")
    
    # Refit on the complete train partition
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)
    
    final_model = HistGradientBoostingRegressor(**best_params)
    final_model.fit(X_scaled, y)
    
    # Get final model's predictions on complete train partition for affine calibration
    train_raw_preds = final_model.predict(X_scaled)
    
    # Fit affine calibration (Linear Regression)
    lr_calib = LinearRegression()
    lr_calib.fit(train_raw_preds.reshape(-1, 1), y)
    
    a = float(lr_calib.intercept_)
    b = float(lr_calib.coef_[0])
    
    logger.info(f"Fitted affine calibration: Acc_hat = clip({a:.5f} + {b:.5f} * HGBR_pred, 0, 1)")
    
    # Create final calibrated predictor
    predictor = CalibratedPredictor(final_scaler, final_model, a, b)
    
    # Get validation predictions with the final calibrated model
    # (using the same train-internal split used for grid search)
    X_val_scaled_final = final_scaler.transform(X_val)
    val_pred_raw = final_model.predict(X_val_scaled_final)
    val_pred_calib = np.clip(a + b * val_pred_raw, 0.0, 1.0)
    
    return predictor, best_params, best_val_mae, val_pred_calib, y_val
