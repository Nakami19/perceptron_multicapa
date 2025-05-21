from matplotlib import pyplot as plt
from activacion import derivada_sigmoide, sigmoidal
from capa import Capa
import numpy as np


class RedNeuro:

    ## Define los valores que ingresa el usuario 
    def __init__(self, num_entradas, num_capas, num_neuronas, num_salidas, tasa_aprendizaje):

        ## Capas de la red
        self.capas =[]

        self.tasa_aprendizaje = tasa_aprendizaje

        ## crear capaz de la red
        for i in range(num_capas + 1):
            ## entrada
            if i ==0:
                self.capas.append(Capa(num_entradas , num_neuronas))
            
            ##capas ocultas
            elif 0 < i < num_capas:
                self.capas.append(Capa(num_neuronas,num_neuronas))
            
            ##capa salida
            else:
                self.capas.append(Capa(num_neuronas, num_salidas))

    
    def propagacion_adelante(self, entrada):
        ## propagar a traves de cada capa
        for capa in self.capas:
            ## Guardar entradas de la capa
            capa.entradas = entrada 

            ## Hacer suma ponderada
            capa.sum_entradas = np.dot(entrada, capa.pesos) + capa.bias

            ## Aplicar funcion de activacion
            capa.salidas = sigmoidal(capa.sum_entradas)

            ## La salida de la capa es la entrada de la siguiente 
            entrada = capa.salidas
        
        return entrada
    
    def backPropagation(self, salida_esperada):
        
        ##Calcular error en capa de salida
        capa_salida = self.capas[len(self.capas)-1]
        error_salida = salida_esperada - capa_salida.salidas

        capa_salida.deltas = derivada_sigmoide(capa_salida.sum_entradas) * error_salida

        ## Propagar error hacia capas ocultas
        ## iterar cantidad de capas menos entrada y capa salida 
        ## empieza capa anterior a la de salida hasta la primera capa oculta
        for i in range(len(self.capas) - 2, -1, -1):
            ##Obtener capa actual y la siguiente (capa que esta antes)
            capa = self.capas[i]
            capa_sig = self.capas[i+1]

            ##Calcular error delta de la capa donde estamos
            delta_propagado = np.dot(capa_sig.pesos, capa_sig.deltas)
            capa.deltas = derivada_sigmoide(capa.sum_entradas) * delta_propagado

    
    def actualizar_pesos(self):

        for capa in self.capas:
            ##Para cada capa se actualiza sus pesos y sesgo
            act_pesos = np.matmul(np.atleast_2d(capa.entradas).T, np.atleast_2d(capa.deltas))
            capa.pesos += self.tasa_aprendizaje * act_pesos
            capa.bias += self.tasa_aprendizaje * capa.deltas.flatten()

    def calcular_precision(self,entradas, salidas_esperadas):
        predicciones = self.predecir(entradas)
        predicciones_binarias = (predicciones >= 0.5).astype(int)  # Convertir a 0 o 1
        salidas = salidas_esperadas.astype(int)
        correctas = np.sum(predicciones_binarias == salidas)
        total = len(salidas_esperadas)
        return correctas / total


    def entrenar(self, entradas_entrenamiento, salidas_esperadas,epocas, epsilon, entradas_prueba, salidas_prueba):
        ## epsilon es margen de error para condicion de parada

        errores =[]
        num_muestras = entradas_entrenamiento.shape[0]
        precisiones_entrenamiento = [] 
        precisiones_prueba = []

        for epoca in range(epocas):
            error_epoca = 0
            indice_entradas = np.arange(num_muestras)

            for i in indice_entradas:
                muestra_entrada = entradas_entrenamiento[i]
                muestras_salida = salidas_esperadas[i]

                salida = self.propagacion_adelante(muestra_entrada)

                ##Calcular error cuadratico medio
                error_muestra = np.mean((muestras_salida - salida) ** 2)
                error_epoca += error_muestra

                ## Retropropagacion y actualizacion de pesos
                self.backPropagation(muestras_salida)
                self.actualizar_pesos()
            
            ##Calcular error promedio de epoca
            error_epoca /= len(indice_entradas)
            errores.append(error_epoca)

            # print(f"Epoca {epoca}, Error = {error_epoca:.5f}")

            # Calcular precisión de entrenamiento
            precision_entrenamiento = self.calcular_precision(entradas_entrenamiento, salidas_esperadas)
            precisiones_entrenamiento.append(precision_entrenamiento * 100)
            precision_prueba = self.calcular_precision(entradas_prueba, salidas_prueba)
            precisiones_prueba.append(precision_prueba * 100)

            print(f"Época {epoca} | Error: {error_epoca:.5f} | "
                  f"Precisión entrenamiento: {precision_entrenamiento:.2%} | "
                  f"Precisión prueba: {precision_prueba:.2%}")

            if (error_epoca < epsilon):
                print(f'Convergencia alcanzada en la epoca {epoca}')
                break
        return errores, precisiones_prueba,precisiones_entrenamiento
    
    def predecir(self, entradas):

        if entradas.ndim == 1: 
            return self.propagacion_adelante(entradas)
        else:
            return np.array([self.propagacion_adelante(entrada) for entrada in entradas])
    
    def graficar(self, precisiones_entrenamiento, precisiones_prueba):
        plt.figure(figsize=(10, 6))
        epocas = range(1, len(precisiones_entrenamiento) + 1)
        plt.plot(epocas, precisiones_entrenamiento, 'b-', label='Precisión entrenamiento')
        plt.plot(epocas, precisiones_prueba, 'r-', label='Precisión prueba')
    
        plt.title('Precisión por época')
        plt.xlabel('Épocas')
        plt.ylabel('Precisión')
        plt.legend()
        plt.grid(True)
        plt.show()




