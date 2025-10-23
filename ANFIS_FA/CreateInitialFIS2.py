import numpy as np
from sklearn.cluster import KMeans

def CreateInitialFIS2(Inputs, Targets, n_clusters=3):
    """
    Create initial Sugeno-style ANFIS structure compatible with evalfis() and TrainFISCost().
    Clusters the input data and generates Gaussian membership parameters and linear consequents.
    """

    X = np.asarray(Inputs, dtype=float)
    Y = np.asarray(Targets, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n_samples, n_inputs = X.shape

    # ---- Step 1: Cluster centers ----
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=4)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_

    # ---- Step 2: Compute sigmas ----
    sigmas = np.zeros_like(centers)
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        if len(cluster_points) > 1:
            sigmas[i] = np.std(cluster_points, axis=0) + 1e-6
        else:
            sigmas[i] = np.full(n_inputs, 0.1)

    # ---- Step 3: Linear consequent parameters ----
    consequents = []
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        cluster_targets = Y[labels == i]
        if len(cluster_points) < n_inputs + 1:
            p = np.zeros(n_inputs)
            c = np.mean(Y)
        else:
            A = np.hstack([cluster_points, np.ones((len(cluster_points), 1))])
            theta, *_ = np.linalg.lstsq(A, cluster_targets, rcond=None)
            p = theta[:-1]
            c = theta[-1]
        consequents.append(list(p) + [c])

    # ---- Step 4: Build FIS dictionary ----
    fis = {"rules": []}
    for i in range(n_clusters):
        rule = {
            "means": centers[i].tolist(),
            "sigmas": sigmas[i].tolist(),
            "consequents": consequents[i]
        }
        fis["rules"].append(rule)

    return fis
