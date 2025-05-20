import numpy as np

class Capa:
## Representa una capa de la neurona
## tiene entradas que recibe y numero de neuronas en la capa
    def __init__(self, num_entradas, num_neuronas):

        ## matriz de pesos de la capa 
        self.pesos = np.random.rand(num_entradas,num_neuronas)

        ##sesgo uno por cada neurona
        self.bias = np.random.rand(num_neuronas)

        ##numero entradas
        self.num_entradas = num_entradas

        ## numero neuronas
        self.num_neuronas = num_neuronas

        ##valores que entran a la capa 
        self.entradas = None

        ## salidas de la capa
        self.salidas = None

        ## deltas para aprendizaje
        self.deltas = None

        ## Suma ponderada antes de activacion
        self.sum_entradas = None