'''Implementacion de K-Means en Python, para hacer un analisis empirico del algoritmo'''

# Autor: Andres Cornejo
# Fuente: https://github.com/python-engineer/MLfromscratch/blob/master/mlfromscratch/kmeans.py

import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

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
        asig += 1 # Una asignacion por la iniciacion del for loop.
        for _ in range (self.K): # For loop para simular el loop de la comprension de listas.
            comp += 1 # Una comparacion por el ciclo del for.
            asig += 1 # La asignacion de cada lista vacia.
        comp += 1 # Una comparacion por el False del for loop.

        # Vector feature que contiene la mediana para cada cluster. (Centroides)
        self.centroids = []
        asig += 1 # Asignacion de la lista de centroides.

    def predict(self, X):
        """Funcion principal del algoritmo, se encarga de inicializar todo, y predecir los centroides optimos iterativamente.
        Donde X es el conjunto de datos de entrada a ser analizado."""
        global asig
        global comp
        start = timer()
        self.X = X
        self.n_samples, self.n_features = X.shape
        asig += 3 # Tres asignaciones de las variables del objeto.

        # Inicializar los centroides.
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        asig += 2 # Asignacion de la variable random_sample_idxs y la asignacion de la funcion random.choice()
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        asig += 1 # Una asignacion por la iniciacion del for loop.
        for _ in random_sample_idxs: # For loop para simular el loop de la comprension de listas.
            asig += 2 # Dos asignaciones, una por self.X[idx] y otra por self.centroids.
            comp += 1 # Una comparacion por cada iteracion del for loop.
        comp += 1 # Una comparacion por el False del for loop.

        # Proceso de optimizacion.
        asig += 1 # Una asignacion por la iniciacion del for loop.
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
            asig += 2
            # Esta comparacion se omite, ya que es solo para graficar, y no se relaciona con la ejecucion del algoritmo.
            if self.plot_steps:
                self.plot()
            # Comparacion por el if self._is_converged.
            comp += 1
            # Revisar si ya convergen los centroides.
            if self._is_converged(centroids_old, self.centroids):
                print("Cantidad de asignaciones:",asig)
                print("Cantidad de comparaciones:", comp)
                # Terminar de contar el tiempo    
                end = timer()
                if not self.plot_steps:
                    # Mostrar cuanto tiempo duro el algoritmo.
                    self._get_time(start, end)
                    # Graficar el resultado del algoritmo.
                    self.plot()
                break

        comp += 1 # Una comparacion por el False del for loop.

        # Clasificar cada muestra como un indice de su cluster.
        # Retornar las etiquetas de los clusters.
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        global asig
        global comp

        labels = np.empty(self.n_samples)
        asig += 2 # Dos asignaciones, una de la variable labels, y otra por llamar a np.empty()
        asig += 2 # Dos asignaciones por iniciaciones de dos for loops.
        for cluster_idx, cluster in enumerate(clusters):
            comp += 1 # Una comparacion por cada iteracion del for loop.
            for sample_idx in cluster:
                comp += 1 # Una comparacion por cada iteracion del for loop.
                labels[sample_idx] = cluster_idx
                asig += 1 # Una asignacion del labels[sample_idx].
            comp += 1 # Una comparacion por el False del for loop.
        comp += 1 # Una comparacion por el False del for loop.
        return labels
    
    def _create_clusters(self, centroids):
        """Crea los clusters vacios, segun la cantidad especificada en el objeto"""
        global asig
        global comp
        # Inicializar los clusters vacios.
        clusters = [[] for _ in range (self.K)]
        asig += 1 # Una asignacion por la iniciacion del for loop.
        for _ in range (self.K):
            asig += 1 # Una asignacion del cluster vacio a la lista.
            comp += 1 # Una comparacion por cada iteracion del for loop.
        comp += 1 # Una comparacion por el False del for loop.

        # Conseguir el centroide mas cercano, y agregar su indice a la lista de clusters.
        asig += 1 # Una asignacion por la iniciacion del for loop.
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
            asig += 2 # Dos asignaciones, una de centroid_idx y otra por el .append().
            comp += 1 # Una comparacion por cada iteracion del for loop
        comp += 1 # Una comparacion por el False del for loop.
        return clusters
    
    def _closest_centroid(self, sample, centroids):
        """Retorna el indice del centroide mas cercano a partir de la muestra actual y los centroides."""
        global asig
        global comp

        # Calcular las distancias de todos los puntos a los centroides.
        distances = [euclidean_distance(sample, point) for point in centroids]
        asig += 1 # Asignacion de variable distances.
        asig += 1 # Una asignacion por la iniciacion del for loop.
        for _ in centroids:
            comp += 1 # Una comparacion por cada iteracion del for loop.
            asig += 1 # Una asignacion por cada llamada euclidiean_distance().
        comp += 1 # Una comparacion por el False del for loop.
            
            
        # Conseguir la distancia minima de la lista de distancias.
        closest_idx = np.argmin(distances)
        asig += 2 # Dos asignaciones, una de la variable closest_idx, y otra por np.argmin().
        return closest_idx

    def _get_centroids(self, clusters):
        global asig
        global comp
        centroids = np.zeros((self.K, self.n_features))
        asig += 2 # Dos asignaciones, una de la variable centroids, y otra por np.zeros().
        asig += 1 # Una asignacion por la iniciacion del for loop.
        for cluster_idx, cluster in enumerate(clusters):
            comp += 1 # Una comparacion por cada iteracion del for loop.
            cluster_mean = np.mean(self.X[cluster], axis=0)
            asig += 2 # Dos asignaciones, una de la variable cluster_mean, y otra por np.mean().
            centroids[cluster_idx] = cluster_mean
            asig += 1 # Una asignacion por centroids[cluster_idx].
        comp += 1 # Una comparacion por el False del for loop.
        return centroids

    def _is_converged(self, centroids_old, centroids):
        global asig
        global comp
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        asig += 1 # Una asignacion por la iniciacion del for loop.
        for _ in range(self.K): # For loop para simular el loop de la comprension de listas.
            comp += 1 # Una comparacion por cada iteracion del for loop.
            asig += 1 # Una asignacion por cada elemento agregado a la lista distances.
        comp += 1 # Una comparacion por el False del for loop.

        comp += 1 # Una comparacion por sum(distances) == 0
        return sum(distances) == 0

    def _get_time(self, start_time, end_time):
        print("Tiempo demorado haciendo el algoritmo")
        print(end_time-start_time)


    def plot(self):
        """Funcion que se encarga de dibujar los datos, clusters y centroides. Al no ser parte del algoritmo, no es analizada."""
        fig, ax = plt.subplots(figsize=(12,8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)
        
        plt.show()