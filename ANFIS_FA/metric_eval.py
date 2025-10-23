import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score

def metrics_eval(y_true, y_pred_probs):
    y_true = np.ravel(y_true)
    y_pred_probs = np.ravel(y_pred_probs)
    y_pred_bin = (y_pred_probs >= 0.5).astype(int)
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred_bin),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred_bin)),
        "roc_auc": roc_auc_score(y_true, y_pred_probs),
        "precision": precision_score(y_true, y_pred_bin, zero_division=0),
        "recall": recall_score(y_true, y_pred_bin, zero_division=0),
        "f1": f1_score(y_true, y_pred_bin, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_pred_probs)
    }
    cm = confusion_matrix(y_true, y_pred_bin)
    return metrics, cm
