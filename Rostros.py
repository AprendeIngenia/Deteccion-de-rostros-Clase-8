# Importamos las librerias
import cv2

# Realizamos VideoCaptura
cap = cv2.VideoCapture(0)

# Leemos el modelo
net = cv2.dnn.readNetFromCaffe("opencv_face_detector.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# Parametros del modelo
# Tamaño
anchonet = 300
altonet = 300
# Valores medios de los canales de color
media = [104, 117, 123]
umbral = 0.7

# Empezamos
while True:
    # Leemos los frames
    ret, frame = cap.read()

    # Si hay error
    if not ret:
        break

    # Realizamos conversion de forma
    frame = cv2.flip(frame, 1)

    # Extraemos info de los frames
    altoframe = frame.shape[0]
    anchoframe = frame.shape[1]

    # Preprocesamos la imagen
    # Images - Factor de escala - tamaño - media de color - Formato de color(BGR-RGB) - Recorte
    blob = cv2.dnn.blobFromImage(frame, 1.0, (anchonet, altonet), media, swapRB = False, crop = False)

    # Corremos el modelo
    net.setInput(blob)
    detecciones = net.forward()

    # Iteramos
    for i in range(detecciones.shape[2]):
        # Extraemos la confianza de esa deteccion
        conf_detect = detecciones[0,0,i,2]
        # Si superamos el umbral (70% de probabilidad de que sea un rostro)
        if conf_detect > umbral:
            # Extraemos las coordenadas
            xmin = int(detecciones[0, 0, i, 3] * anchoframe)
            ymin = int(detecciones[0, 0, i, 4] * altoframe)
            xmax = int(detecciones[0, 0, i, 5] * anchoframe)
            ymax = int(detecciones[0, 0, i, 6] * altoframe)

            # Dibujamos el rectangulo
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
            # Texto que vamos a mostrar
            label = "Confianza de deteccion: %.4f" % conf_detect
            # Tamaño del fondo del label
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # Colocamos fondo al texto
            cv2.rectangle(frame, (xmin, ymin - label_size[1]), (xmin + label_size[0], ymin + base_line),
                          (0,0,0), cv2.FILLED)
            # Colocamps el texto
            cv2.putText(frame, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    cv2.imshow("DETECCION DE ROSTROS", frame)

    t = cv2.waitKey(1)
    if t == 27:
        break

cv2.destroyAllWindows()
cap.release()
