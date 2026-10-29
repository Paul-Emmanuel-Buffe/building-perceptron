"""Le perceptron fait deux choses :
Prédire : predict calcule z, puis ŷ.
Apprendre : fit corrige w et b quand il se trompe."""

import numpy as np


class Perceptron:
    
    """ranger les réglages dans l’objet :

    learning_rate est le η de la formule.
    n_iter est le nombre maximum d’époques, c’est-à-dire de passages sur toutes les données.
    weights et bias valent None pour l’instant, car le perceptron n’a encore rien appris.
    errors est une liste vide où on notera le nombre d’erreurs à chaque époque, pour faire le graphique.""" 
    
    def __init__(self, learning_rate=0.01, n_iter=50):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
        self.errors = []
        



    """L’intérêt est simple : fit trouve les bons poids et les garde en mémoire dans l’objet. Ensuite, predict réutilise ces mêmes poids."""
    
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # on commence avec des poids a 0
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.errors = []

        for i in range(self.n_iter):
            nb_erreurs = 0

            for j in range(n_samples):
                y_pred = self.predict(X[j])
                erreur = y[j] - y_pred

                # mise a jour des poids et du biais
                self.weights = self.weights + self.learning_rate * erreur * X[j]
                self.bias = self.bias + self.learning_rate * erreur

                if erreur != 0:
                    nb_erreurs += 1

            self.errors.append(nb_erreurs)

            # si il n'y a plus d'erreurs on arrete
            if nb_erreurs == 0:
                break
    
        
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias #"""C’est exactement z = w·x + b. np.dot fait « multiplier terme à terme puis additionner »"""
        return np.where(z >= 0, 1, 0)