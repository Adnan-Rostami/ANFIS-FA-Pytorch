import numpy as np

def gaussmf(x, mean, sigma):
    """Gaussian membership function."""
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2)

def evalfis(X, fis):
    """
    Evaluate Sugeno-type ANFIS output.
    Compatible with CreateInitialFIS2 output structure.
    """
    X = np.atleast_2d(X)
    n_samples, n_inputs = X.shape
    n_rules = len(fis["rules"])

    w = np.zeros((n_samples, n_rules))
    y_rule = np.zeros((n_samples, n_rules))

    for i, rule in enumerate(fis["rules"]):
        means = np.array(rule["means"])
        sigmas = np.array(rule["sigmas"])
        # --- گوسین ممبرشیپ ---
        mu = np.ones(n_samples)
        for j in range(n_inputs):
            mu *= gaussmf(X[:, j], means[j], sigmas[j])
        w[:, i] = mu

        # --- خروجی خطی قاعده ---
        p = np.array(rule["consequents"][:-1])
        c = rule["consequents"][-1]
        y_rule[:, i] = X @ p + c



    # نرمال‌سازی وزن‌ها
    w_sum = np.sum(w, axis=1, keepdims=True)
    w_norm = np.divide(w, w_sum, out=np.zeros_like(w), where=w_sum != 0)

    y_pred = np.sum(w_norm * y_rule, axis=1)
    return y_pred
