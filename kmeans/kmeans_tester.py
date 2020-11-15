"""Archivo para realizar pruebas y analisis del algoritmo kmeans."""
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from main import KMeans

SAMPLE_SIZE = 200 # Cantidad de elementos en la muestra.

# Crear arreglo base.
arr = []
invArr = []
# Llenar el arreglo base con datos ascendentes.
for i in range(0, SAMPLE_SIZE):
    arr.append([i, i])
    invArr.append([SAMPLE_SIZE-i, i])

# Crear el conjunto de datos ascendente.
ascendente = np.ndarray(shape=(SAMPLE_SIZE, 2), buffer=np.array(arr), dtype=int)
# Crear el conjunto de datos descendente.
descendente = np.ndarray(shape=(SAMPLE_SIZE, 2), buffer=np.array(invArr), dtype=int)
# El random_state siempre es igual para tener resultados reproducibles, para la medicion empirica.
# X, y = make_blobs(centers=3, n_samples=1000, n_features=2, shuffle=True, random_state=133742069)

# Crear el conjunto de datos aleatorios utilizando sklearn.
X, y = make_blobs(centers=3, n_samples=SAMPLE_SIZE, n_features=2, shuffle=True)

print("Dimensiones del arreglo: ", X.shape)

# Generar clusters no repetidos. (np.unique())
clusters = len(np.unique(y))
print("Cantidad de clusters: ", clusters)

# Crear el objeto tipo K-Mean y correr el algoritmo.
k = KMeans(K=clusters, max_iters=300, plot_steps=False, plot_final=True)
y_pred = k.predict(descendente)
