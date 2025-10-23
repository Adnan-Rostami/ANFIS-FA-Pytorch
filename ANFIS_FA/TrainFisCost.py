# ===============================================================
# TrainFISCost.py | محاسبه‌ی هزینه‌ی مدل فازی برای Firefly
# ===============================================================

import numpy as np
from evalfis import evalfis

def set_fis_params(fis, x):
    """اعمال متغیرهای Firefly روی پارامترهای FIS"""
    idx = 0
    if not isinstance(fis, dict) or "rules" not in fis:
        raise ValueError("❌ FIS object malformed or missing 'rules' key.")
    for rule in fis["rules"]:
        for j in range(len(rule["means"])):
            if idx < len(x):
                rule["means"][j] += 0.05 * x[idx]
                idx += 1
        for j in range(len(rule["sigmas"])):
            if idx < len(x):
                rule["sigmas"][j] *= (1 + 0.05 * x[idx])
                idx += 1
    return fis


def TrainFISCost(x, data):
    """
    محاسبه‌ی هزینه‌ی FIS با معیار Mixed RMSE و (1-F1)
    x     : آرایه‌ پارامترهای کرم شب‌تاب
    data  : {'Inputs': X, 'Targets': y, 'fis': ساختار فازی}
    خروجی:
        - RMSE (به‌عنوان cost)
        - info شامل fis, MSE, RMSE, MAE, SSE, F1, Cost
    """

    MinAbs = 1e-5
    x = np.array(x, dtype=float)
    x[np.abs(x) < MinAbs] = MinAbs * np.sign(x[np.abs(x) < MinAbs])

    # --- ساخت FIS بر اساس پارامترها ---
    fis = data['fis']
    fis = set_fis_params(fis, x)

    # --- پیش‌بینی و محاسبات ---
    X = np.array(data["Inputs"], dtype=float)
    T = np.array(data["Targets"], dtype=float).flatten()
    Y = evalfis(X, fis)

    e = T - Y
    MSE = np.mean(e ** 2)
    RMSE = np.sqrt(MSE)
    MAE = np.mean(np.abs(e))
    SSE = np.sum(e ** 2)

    # --- Classification-aware cost ---
    y_bin = (Y >= 0).astype(int)
    t_bin = (T >= 0).astype(int)
    TP = np.sum((y_bin == 1) & (t_bin == 1))
    FP = np.sum((y_bin == 1) & (t_bin == 0))
    FN = np.sum((y_bin == 0) & (t_bin == 1))
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    # ترکیب RMSE و (1-F1)
    alpha, beta = 1, 1
    cost = alpha * RMSE + beta * (1 - f1)

    out = {
        "fis": fis,
        "MSE": MSE,
        "RMSE": RMSE,
        "MAE": MAE,
        "SSE": SSE,
        "F1": f1,
        "Cost": cost
    }

    return RMSE, out
