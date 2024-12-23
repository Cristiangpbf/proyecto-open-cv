import cv2
import os
import numpy as np
from datetime import datetime
import time

dataPath = 'recursos_reconocimiento_facial/Data'  # Cambia a la ruta donde hayas almacenado Data
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

labels = []
facesData = []
label = 0

# Inicio de la medición del tiempo
start_time = time.time()

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Leyendo las imágenes')

    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath + '/' + fileName, 0))
        # image = cv2.imread(personPath+'/'+fileName,0)
        # cv2.imshow('image',image)
        # cv2.waitKey(10)
    label = label + 1

print('labels= ', labels)
print('Número de etiquetas 0: ', np.count_nonzero(np.array(labels) == 0))
print('Número de etiquetas 1: ', np.count_nonzero(np.array(labels) == 1))
print('Número de etiquetas 2: ', np.count_nonzero(np.array(labels) == 2))
print('Número de etiquetas 3: ', np.count_nonzero(np.array(labels) == 3))

# Métodos para entrenar el reconocedor
# face_recognizer = cv2.face.EigenFaceRecognizer_create()
# face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Entrenando el reconocedor de rostros
print("Entrenando...")
print(f"Inicio Entrenamiento: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}")
face_recognizer.train(facesData, np.array(labels))
print(f"Termino Entrenamiento: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}")

# Almacenando el modelo obtenido
# face_recognizer.write('modeloEigenFace.xml')
# face_recognizer.write('modeloFisherFace.xml')
face_recognizer.write('modeloLBPHFace.xml')
print(f"Modelo almacenado... {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}")

# Fin de la medición del tiempo
end_time = time.time()
execution_time = end_time - start_time

# Convertir tiempo total en minutos y segundos
minutes = int(execution_time // 60)
seconds = execution_time % 60

# Mostrar el tiempo en formato legible
print(f"Total tiempo: {execution_time:.2f} segundos")
print(f"Tiempo total: {minutes} minutos y {seconds:.2f} segundos")



# Imprimir la fecha y hora formateada
print(f"Inicio Entrenamiento: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}")