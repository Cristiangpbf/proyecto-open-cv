import cv2
import os
import imutils
import datetime

start_time = datetime.datetime.now()
print(f'Iniciando el proceso: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')

personName = 'Ambar'
dataPath = 'C:/imagenes_reconocimiento_facial/Data'
personPath = dataPath + '/' + personName

if not os.path.exists(personPath):
    print('Carpeta creada: ', personPath)
    os.makedirs(personPath)

# cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
# Descomentar para capturar registros de un video.
cap = cv2.VideoCapture('C:/imagenes_reconocimiento_facial/Videos/'+personName+'.mp4')

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades +
                                    'haarcascade_frontalface_default.xml')
count = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personPath + '/rotro_{}.jpg'.format(count), rostro)
        count = count + 1

    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27 or count >= 300:
        break

cap.release()
cv2.destroyAllWindows()

# Al final del código
end_time = datetime.datetime.now()
print(f'Proceso terminado: {end_time.strftime("%Y-%m-%d %H:%M:%S")}')
