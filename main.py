import sys
import numpy as np
import tkinter as tk
from tkinter import filedialog
from capa import Capa
from red_neuro import RedNeuro

def menu_principal():
    print("Perceptrón Multicapa")
    print("1. Crear nuevo perceptrón")
    print("2. Cargar perceptrón desde archivo")
    print("3. Salir")
    return input("Seleccione opción (1/2/3): ").strip()

def menu_red():
    print("\nSeleccione una opción:")
    print("1. Ejecutar red (vector por teclado)")
    print("2. Ejecutar red (archivo de prueba .txt)")
    print("3. Seguir entrenando")
    print("4. Guardar red en archivo")
    print("5. Mostrar gráfico de precisiones")
    print("6. Salir")
    return input("Seleccione opción (1/2/3/4/5): ").strip()


def cargar_datos(nombre_archivo):
    try:
        ## obtengo datos del archivo como un arreglo numpy
        datos = np.loadtxt(nombre_archivo, delimiter=',', dtype=float)
        if datos.ndim == 1:
            return datos.reshape(-1,1)
        return datos
    except Exception as e:
        print(f"Error al cargar {nombre_archivo}: {e}")
        sys.exit(1)

def seleccionar_archivo(mensaje="Seleccione un archivo"):
    root = tk.Tk()
    root.withdraw() 
    root.attributes('-topmost', True) 
    archivo = filedialog.askopenfilename(title=mensaje, filetypes=[("Text files", "*.txt")])
    root.destroy()
    return archivo

def ejecutar_vector(red):
    entrada_str = input("Ingrese valores separados por comas (ej: 1.0,0.5,3.2): ")

    if ',' not in entrada_str:
        print("Debe ingresar al menos dos valores separados por comas.")
        return
    
    valores = entrada_str.strip().split(',')
    if len(valores) < 2:
        print("Debe ingresar al menos dos valores.")
        return
    
    num_entradas = red.capas[0].num_entradas

    if len(valores) != num_entradas:
        print(f"Número incorrecto de entradas. Se esperaban {num_entradas}, pero se ingresaron {len(valores)}.")
        return
    
    try:
        vector = np.array([float(x) for x in entrada_str.split(',')])
    except:
        print("Formato inválido.")
        return
    salida = red.predecir(vector)
    print(f"Salida: {salida}")

def cargar_red(archivo, tasa_aprendizaje):
    try:
        with open(archivo, 'r') as f:
            ## obtengo todas las lineas del archivo
            lineas = f.readlines()

        ##Primera linea es el numero de capas
        num_capas = int(lineas[0].strip())

        ##Creo una red vacia donde se iran guardando los datos
        red = RedNeuro(0, 0, 0, 0, tasa_aprendizaje) 

        ##indice para ir recorriendo lineas del archivo
        idx_lineas = 1

        ##reinicio lista de capas de red porsia 
        red.capas = []

        ##Por cada capa
        for i in range(num_capas):
            ## obtengo de la linea una lista de enteros 
            ## estos seran la cantidad de entradas y salidas
            dims = list(map(int, lineas[idx_lineas].strip().split(',')))

            ## aumento idz para ir a siguiente linea 
            idx_lineas += 1

            ##pesos de la capa
            pesos = []

            ## Se recorre cada entrada donde cada una tiene su peso
            for y in range(dims[0]):
                ##obtengo de la linea una lista de enteros 
                ##estos seran los pesos
                fila = list(map(float, lineas[idx_lineas].strip().split(',')))
                ##agrego los pesos a la lista
                pesos.append(fila)
                ##paso a siguiente linea
                idx_lineas += 1
            
            # leer bias 
            bias = list(map(float, lineas[idx_lineas].strip().split(',')))
            idx_lineas += 1
            
            ##Se crea la capa
            ## dim[0] entradas, dim[1] salidas
            capa = Capa(dims[0], dims[1])
            ## guardo los pesos del archivo en la capa
            capa.pesos = np.array(pesos)
            ## guardo el bias
            capa.bias = np.array(bias)
            ##agrego la capa a la red
            red.capas.append(capa)

        return red

    except Exception as e:
        print(f"Error al cargar la red: {e}")
        return None


def guardar_red(red, archivo):
    try:
        with open(archivo, 'w') as f:
            f.write(f"{len(red.capas)}\n")  # Número total de capas

            ## recorro cada capa dentro de la red
            for capa in red.capas:
                ## obtengo numero de entradas y salidas en la capa
                filas, cols = capa.pesos.shape
                f.write(f"{filas},{cols}\n")
                ## recorro cada fila de la matriz de pesos
                for fila in capa.pesos:
                    ##coloco los pesos de cada fila en un string para guardarlos
                    f.write(','.join(map(str, fila)) + '\n')

                ##guardo el bias al final
                f.write(','.join(map(str, capa.bias)) + '\n')
        print(f"Red guardada exitosamente en '{archivo}'")
    except Exception as e:
        print(f"Error al guardar la red: {e}")

def continuar_entrenando(red):
    ##Pido los datos de entrenamiento
    archivo_ent = seleccionar_archivo("Archivo de entrenamiento (.txt): ")
    archivo_sal_ent = seleccionar_archivo("Archivo de salidas de entrenamiento esperadas (.txt): ")
    archivo_pru = seleccionar_archivo("Archivo de prueba (.txt): ")
    archivo_sal_pru = seleccionar_archivo("Archivo de salidas esperadas (prueba .txt): ")
    epsilon = float(input("Error aceptable: "))
    try:
        ##obtengo los datos
        X_ent = cargar_datos(archivo_ent)
        Y_ent = cargar_datos(archivo_sal_ent)
        X_pru = cargar_datos(archivo_pru)
        Y_pru = cargar_datos(archivo_sal_pru)
    except:
        print("Error al cargar los datos. Verifique los archivos e intente nuevamente.")
        return
    ##entreno la red
    epocas = int(input("Número de épocas adicionales: "))
    errores,precisiones_prueba,precisiones_entrenamiento = red.entrenar(entradas_entrenamiento= X_ent, salidas_esperadas=Y_ent, epocas=epocas, epsilon=epsilon, entradas_prueba=X_pru, salidas_prueba=Y_pru)
    print('Red entrenada con éxito')
    return errores, precisiones_prueba,precisiones_entrenamiento

def main():

    ##variables donde guardare precisiones para grafica
    precisiones_entrenamiento = None
    precisiones_prueba = None

    while True:
        opcion = menu_principal()
        ## obtener parametros del usuario
        if opcion == '1':
            num_entradas = int(input("Número de neuronas de entrada: "))
            num_salidas = int(input("Número de neuronas de salida: "))
            num_capas = int(input("Número de capas ocultas: "))
            if num_capas > 0:
                num_neuronas = int(input("Número de neuronas por capa oculta: "))
            else:
                num_neuronas = 0

            archivo_ent = seleccionar_archivo("Seleccione el archivo de entrenamiento (.txt)")
            archivo_sal_ent = seleccionar_archivo("Seleccione archivo de salidas de entrenamiento esperadas (.txt)")
            archivo_pru = seleccionar_archivo("Seleccione archivo de prueba (.txt)")
            archivo_sal_pru = seleccionar_archivo("Seleccione archivo de salidas esperadas de prueba (.txt)")
            epocas = int(input("Número de épocas: "))

            X_ent = cargar_datos(archivo_ent)
            Y_ent = cargar_datos(archivo_sal_ent)
            X_pru = cargar_datos(archivo_pru)
            Y_pru = cargar_datos(archivo_sal_pru)

            ##Creo red con datos del usuario e inicio entrenamiento
            red = RedNeuro(num_entradas, num_capas, num_neuronas, num_salidas, tasa_aprendizaje= 0.5)
            errores,precisiones_prueba,precisiones_entrenamiento = red.entrenar(entradas_entrenamiento= X_ent, salidas_esperadas=Y_ent, epocas=epocas, epsilon=0.02, entradas_prueba=X_pru, salidas_prueba=Y_pru)
            print('Red entrenada con éxito')
            break

        elif opcion=='2': 
            archivo_cargar = seleccionar_archivo("Archivo con red guardada (.txt): ")
            red = cargar_red(archivo_cargar, tasa_aprendizaje=0.5)
            if red is None:
                print("No se pudo cargar la red. Verifique el archivo e intente nuevamente.")
                continue
            print('Red cargada con éxito')
            break
        elif opcion=='3':
            print("Saliendo...")
            sys.exit(1)
        else:
            print("Opción inválida.")



    while True:
        opcion2 = menu_red()
        if opcion2 == '1':
            ## Ejecutar vector por teclado
            ejecutar_vector(red)
        elif opcion2 == '2':
            ## Ejecutar datos de un archivo
            archivo = seleccionar_archivo("Nombre de archivo de prueba (.txt, valores separados por comas): ")
            entradas = cargar_datos(archivo)
            for i in range(len(entradas)):
                prediccion = red.predecir(entradas[i])
                entrada_formateada = ', '.join(f"{v:.1f}" for v in entradas[i])
                print(f"Entrada [{entrada_formateada}] => Predicción: {prediccion[0]:.5f}")
            
        elif opcion2 == '3':
            ## Seguir entrenando
            errores,precisiones_prueba,precisiones_entrenamiento = continuar_entrenando(red)
        elif opcion2 == '4':
            ## Guardar red
            archivo_guardar = seleccionar_archivo("Archivo para guardar la red (.txt): ")
            guardar_red(red, archivo_guardar)
        
        elif opcion2 == '5':
            if (precisiones_entrenamiento == None or precisiones_prueba==None):
                print('Se debe realizar el entrenamiento primero')
                continue
            red.graficar(precisiones_entrenamiento=precisiones_entrenamiento, precisiones_prueba=precisiones_prueba)
            
        elif opcion2 == '6':
            print("Saliendo...")
            break
        else:
            print("Opción inválida.")


main()