### Funcion sigmoidal ###
import numpy as np


def sigmoidal(num):
    resultado = 1/(1+np.exp(-num))
    return resultado

def derivada_sigmoide(num):
    resultado = sigmoidal (num)

    deriv = resultado * (1 - resultado)

    return deriv