'''Implementacion de K-Means en Python, para hacer un analisis empirico del algoritmo'''

# Autor: Andres Cornejo
# Fuente: https://github.com/python-engineer/MLfromscratch/blob/master/mlfromscratch/kmeans.py

import numpy as np
import matplotlib.pyplot as plt

asig = 0 # Variable testigo que almacena las asignaciones.
comp = 0 # Variable testigo que almacena las comparaciones.

np.random.seed(42) # Utilizar siempre la misma semilla, para poder tener resultados reproducibles.

def euclidean_distance(x1,x2):
    """Funcion que calcula la distancia euclideana entre dos puntos"""
    global asig
    asig += 2 # Dos asignaciones, por la funcion de suma y la funcion de raiz cuadrada.
    return np.sqrt(np.sum((x2-x1)**2))

class KMeans:

    def __init__(self, K=5, max_iters=100, plot_steps=False):
        global asig 
        global comp
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        asig += 3 # Tres asignaciones de las variables del objeto.

        # Lista de indices para cada cluster.
        # Inicializar una lista vacia para cada cluster.
        self.cluster = [[] for _ in range (self.K)]
        for _ in range (self.K):
            comp += 1 # Una comparacion por el ciclo del for.
            asig += 1 # La asignacion de cada lista vacia.
        # Vector feature que contiene la mediana para cada cluster. (Centroides)
        self.centroids = []
        asig += 1 # Asignacion de vector de centroides.

    def predict(self, X):
        """Funcion principal del algoritmo, se encarga de inicializar todo, y predecir los centroides optimos iterativamente"""
        global asig
        global comp
        self.X = X
        self.n_samples, self.n_features = X.shape
        asig += 3 # Tres asignaciones de las variables del objeto.

        # Inicializar los centroides.
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        asig += 2 # Asignacion de la variable random_sample_idxs y la asignacion de la funcion random.choice()
        self.centroids = [self.X[idx] for idx in random_sample_idxs]
        for _ in random_sample_idxs:
            asig += 2 # Dos asignaciones, una por self.X[idx] y otra por self.centroids.
            comp += 1 # Una comparacion por cada iteracion del for loop.

        # Proceso de optimizacion.
        for _ in range(self.max_iters):
            # Actualizar cada cluster.
            self.clusters = self._create_clusters(self.centroids)
            asig += 1 # Asignacion de self.clusters.
            comp += 1 # Comparacion del if.
            if self.plot_steps:
                self.plot()

            # Actualizar los centroides viejos y conseguir los nuevos.
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            # Dos asignaciones de centroids_old y self.centroids.
            asig += 1
            # Comparacion por el if.
            comp += 1
            if self.plot_steps:
                self.plot()
            # Comparacion por el if.
            comp += 1
            # Revisar si ya convergen los centroides.
            if self._is_converged(centroids_old, self.centroids):
                print("Cantidad de asignaciones:",asig)
                print("Cantidad de comparaciones:", comp)
                self.plot()
                break

        # Clasificar cada muestra como un indice de su cluster.
        # Retornar las etiquetas de los clusters.
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels
    
    def _create_clusters(self, centroids):
        clusters = [[] for _ in range (self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters
    
    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def plot(self):
        """Funcion que se encarga de dibujar los datos, clusters y centroides. Al no ser parte del algoritmo, no es analizada."""
        fig, ax = plt.subplots(figsize=(12,8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)
        
        plt.show()