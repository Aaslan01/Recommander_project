
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score

def evaluate_classification(y_true, y_pred, k=10):
    y_pred_bin = (y_pred > 0.5).astype(int)
    precision = precision_score(y_true, y_pred_bin)
    recall = recall_score(y_true, y_pred_bin)
    f1 = f1_score(y_true, y_pred_bin)
    return precision, recall, f1

def evaluate_regression(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae
