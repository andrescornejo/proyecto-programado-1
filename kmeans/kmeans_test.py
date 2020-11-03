import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from main import KMeans
from timeit import default_timer as timer

X, y = make_blobs(centers=3, n_samples=2000, n_features=2, shuffle=True, random_state=42)
print(X.shape)

# Generar clusters no repetidos.
clusters = len(np.unique(y))
print(clusters)

# Crear el objeto tipo K-Mean y correr el algoritmo.
k = KMeans(K=clusters, max_iters=200, plot_steps=False)
start = timer()
y_pred = k.predict(X)
end = timer()
print("Tiempo demorado haciendo el algoritmo")
print(end-start)
