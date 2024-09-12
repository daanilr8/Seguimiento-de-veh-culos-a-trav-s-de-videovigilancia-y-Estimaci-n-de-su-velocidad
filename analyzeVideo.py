from ultralytics import YOLO

import cv2
import os
import json
import math
import numpy as np
import imageio
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torchvision

from PIL import Image,ImageDraw,ImageFont
import PIL.ExifTags

model = YOLO("yolov8n.pt")

distancia_real_coche = []
distancia_estimada_coche = []
distancia_real_furgoneta = []
distancia_estimada_furgoneta = []
distancia_real_camion = []
distancia_estimada_camion = []
t=0
realPositionMin = []
realPositionMed = []
realPositionMax = []
positionErrorMin = []
positionErrorMed = []
positionErrorMax = []
realDistance = 0
difVel = 0
difVelReal = 0
last_real_distance = 0
realx1 = 0
realx2 = 0
realy1 = 0
realy2 = 0
cx = 0
cy = 0
cz = 0
estimated_velocity = []
real_velocity = []
error_velocity = []
positionError = []
distance_to_center = []
estimated_distance = []
realPosition = []
numObjectinJson = 300
last_distance = 0
last_id = 200
count_Id = 0
images = []
ruta_carpeta = os.path.join(os.getcwd(), 'videos','videos2','video21')
ruta_carpeta_cam = os.path.join(os.getcwd(), 'videos','videos2','000080','cam03')
nombre_fichero_json = os.path.join(os.getcwd(), 'videos','videos2',"000080.json")
ruta_graficas = os.path.join(os.getcwd(), 'videos','videos1','video22Reso')


'''
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        analyze_image(frame)

    cap.release()
'''


def analyze_video(video_path):
        loadImages()
        NumFramePixelsHeight = 1024
        NumFramePixelsWidth = 1920
        count_frames = 0
        results = model.track(source=video_path, conf=0.3, iou=0.5)
        for result in results:
            distance = 2000
            frame_id_buscado = frameid(count_frames)
            finalObjectPixelSizeHeight = 0
            finalObjectPixelSizeWidth = 0
            tipoVehiculo = ""
            num_objects = result.boxes.xyxy.shape[0]
            # Analizar bounding box
            ids = result.boxes.id
            if ids is not None:
                for i in range(len(ids)):
                    id = result.boxes.id[i]
                    x1,y1,x2,y2 = result.boxes.xyxy[i]
                    etiqueta = result.names.get(int(result.boxes.cls[i]))
                    if((x2-x1)*(y2-y1) > 3500):
                        d = abs(((x2+x1)/2)-960)
                        if (d < distance):
                            distance = d
                            finalx1 = x1
                            finalx2 = x2
                            finaly1 = y1
                            finaly2 = y2
                            finalId = id
                            finalEtiqueta = etiqueta
                #print("El id final es : " + str(finalId) +  "y el frame es :" + str(count_frames))
                if(finalEtiqueta == "car"):
                    if((finalx2-finalx1) > 0.8*(finaly2-finaly1)):
                        tipoVehiculo = "Coche"
                    else:
                        tipoVehiculo = "Furgoneta"
                elif(finalEtiqueta == "truck"):
                    tipoVehiculo = "Camión"
                else:
                    tipoVehiculo = "Bus"

            DistanceToObject(x1,x2,y1,y2,finalId,count_frames,frame_id_buscado,tipoVehiculo)
            count_frames = count_frames+1
            # print("Distance " + str(distance))
            # print("FinalObjectPixelSizeWidth " + str(x1) + str(x2))
            # print("FinalObjectPixelSizeHeight " + str(y1) + str(y2))
        loadVideo()
        mostrar_graficas()


def DistanceToObject(finalx1,finalx2,finaly1,finaly2,finalId,count_frames,frame_id_buscado,tipoVehiculo):

    NumFramePixelsHeight = 1024
    NumFramePixelsWidth = 1920

    ObjectPixelSizeWidth = finalx2-finalx1
    ObjectPixelSizeHeight = finaly2-finaly1
    # Compute the angle of view of the camera (in radians) in both directions
    AngleOfViewWidth = 120 * (math.pi/180)
    AngleOfViewHeight = 37 * 2 * (math.pi/180)
    # Compute the angle of view associated to a single pixel (in radians) in both directions
    PixelAngleOfViewWidth=AngleOfViewWidth/NumFramePixelsWidth
    PixelAngleOfViewHeight=AngleOfViewHeight/NumFramePixelsHeight
    # Real size of the object in meters
    if(tipoVehiculo == "Coche"):
        RealSizeObjectWidth=1.8
        RealSizeObjectHeight=1.5
    elif(tipoVehiculo == "Furgoneta"):
        RealSizeObjectWidth= 1.9
        RealSizeObjectHeight= 2
    else:
        RealSizeObjectWidth= 2.5
        RealSizeObjectHeight= 4

    # Compute the angle of view associated to the object (in radians) in both directions
    ObjectAngleOfViewWidth=PixelAngleOfViewWidth*ObjectPixelSizeWidth
    ObjectAngleOfViewHeight=PixelAngleOfViewHeight*ObjectPixelSizeHeight
    # Compute the real distance to the object (in meters) according to the apparent sizes in both directions
    DistanceToObject=(RealSizeObjectWidth/2) / (math.tan(ObjectAngleOfViewWidth/2))
    #print(DistanceToObject)
    DistanceToObject2=(RealSizeObjectHeight/2) / (math.tan(ObjectAngleOfViewHeight/2))
    DistanceToObjectMedium = ((DistanceToObject+DistanceToObject2)/2)
    #print(DistanceToObject2)
    realDistance,trueAnnos = real_distance(finalx1,finalx2,finaly1,finaly2,frame_id_buscado,DistanceToObject2,tipoVehiculo)
    getVelocity(finalx1,finalx2,finaly1,finaly2,DistanceToObject2,finalId,count_frames,frame_id_buscado,trueAnnos,realDistance,tipoVehiculo)

def getVelocity(finalx1,finalx2,finaly1,finaly2,distance,finalId,count_frames,frame_id_buscado,trueAnnos,realDistance,tipoVehiculo):

    global difVel
    global difVelReal
    global last_distance
    global last_real_distance
    global last_id
    global real_velocity
    global estimated_velocity
    global error_velocity
    global t


    t = t+1
    IntFinalId = int(finalId)
    if(trueAnnos):
        if(IntFinalId == last_id):
            difVelReal = (realDistance-last_real_distance)/(0.1*t)
            difVel = (distance-last_distance)/(0.1*t)
            last_real_distance = realDistance
            last_distance = distance
            t = 0
        else:
            last_id = IntFinalId
        estimated_velocity.append(difVel)
        real_velocity.append(difVelReal)
        error_velocity.append(abs(abs(difVel)-abs(difVelReal)))
    plotImage(finalx1,finalx2,finaly1,finaly2,distance,last_distance,finalId,count_frames,difVel,realDistance,difVelReal,tipoVehiculo,trueAnnos)


def loadImages():
    
    global images
    global ruta_carpeta

    # Obtén la lista de nombres de archivo de las imágenes en la carpeta
    lista_archivos = os.listdir(ruta_carpeta)

    # Filtrar solo los archivos de imagen (por ejemplo, con extensión .jpg)
    for archivo in lista_archivos:
         # Construye la ruta completa a cada imagen
        ruta_imagen = os.path.join(ruta_carpeta, archivo)

        # Abre la imagen utilizando PIL
        img = Image.open(ruta_imagen)

        # Agrega la imagen a la lista
        images.append(img)


def plotImage(finalx1,finalx2,finaly1,finaly2,distance,last_distance,finalId,count_frames,difVel,realDistance,difVelReal,tipoVehiculo,trueAnnos):

    global images
    global ruta_carpeta
    global list_count_frames
    global realPosition
    global estimated_distance
    img = images[count_frames]
    # Crear un objeto ImageDraw para dibujar en la imagen
    draw = ImageDraw.Draw(img)
        # Dibujar en la imagen
    if(distance > last_distance):
        #Acelerando
        draw.rectangle([finalx1, finaly1, finalx2, finaly2], outline='green', width=8)
    else:
        #Frenando
        draw.rectangle([finalx1, finaly1, finalx2, finaly2], outline='red', width=8)

    roundDistance = round(distance,3)
    font = ImageFont.truetype("arial.ttf", 50)
    texto_Id = "Identificador: " + str(int(finalId))
    texto_distance = "Distancia estimada: " + str(roundDistance)
    texto_velocity = "Diferencia de velocidad estimada: " + str(round(difVel,3))
    texto_etiqueta = "Tipo de vehículo: " + tipoVehiculo
    texto_real_distance = "Distancia real: " + str(round(realDistance,3))
    texto_real_velocity = "Diferencia de velocidad real: " + str(round(difVelReal,3))
    color_texto = (255,0,0)
    color_negro = (0,0,0)
    draw.text((20,30),texto_Id,fill=color_texto,font=font)
    draw.text((20,80),texto_etiqueta,fill=color_texto,font=font)
    draw.text((20,130),texto_distance,fill=color_texto,font=font)
    draw.text((20,180),texto_real_distance,fill=color_texto,font=font)
    draw.text((20,230),texto_velocity,fill=color_texto,font=font)
    draw.text((20,280),texto_real_velocity,fill=color_texto,font=font)

    images[count_frames] = img

def loadVideo():

    global images
    nombre_video = "video21Coches.mp4"
    fps = 5.0
    imageio.mimsave(nombre_video,images,fps=fps)

def frameid(count_frames):

    global ruta_carpeta

    lista_archivos = os.listdir(ruta_carpeta)
    
    return lista_archivos[count_frames]

def real_distance(finalx1,finalx2,finaly1,finaly2,frame_id_buscado,estimatedDistance,tipoVehiculo):
    global nombre_fichero_json
    global ruta_carpeta
    global numObjectinJson
    global realPosition
    global distance_to_center
    global positionError
    global estimated_distance
    global cx
    global cy
    global cz
    global positionErrorMin
    global realPositionMin
    global positionErrorMed
    global realPositionMed
    global positionErrorMax
    global realPositionMax
    global distancia_real_coche
    global distancia_estimada_coche
    global distancia_real_furgoneta
    global distancia_estimada_furgoneta
    global distancia_real_camion
    global distancia_estimada_camion

    B = [finalx1,finaly1,finalx2,finaly2]
    tensorB = torch.tensor([B])
    count = 0
    distance = 3000
    trueAnnos = False


    with open(nombre_fichero_json,'r') as archivo_json:
        datos = json.load(archivo_json)


    frame_id_buscado_extension = frame_id_buscado[:-4]


    # Buscar el frame_id en la lista de frames
    for frame in datos["frames"]:
        count = count+1
        if frame["frame_id"] == frame_id_buscado_extension:
            if len(datos["frames"][count-1]) == 4:
                A = []
                for i in range(len(datos["frames"][count-1]["annos"]["boxes_2d"]["cam03"])):
                    x1 = datos["frames"][count-1]["annos"]["boxes_2d"]["cam03"][i][0]
                    y1 = datos["frames"][count-1]["annos"]["boxes_2d"]["cam03"][i][1]
                    x2 = datos["frames"][count-1]["annos"]["boxes_2d"]["cam03"][i][2]
                    y2 = datos["frames"][count-1]["annos"]["boxes_2d"]["cam03"][i][3]
                    new_bb = [x1,y1,x2,y2]
                    if(x1 == -1):
                        new_bb_1 = [0,0,0,0]
                        A.append(new_bb_1)
                    else:
                        A.append(new_bb)
                tensorA = torch.tensor(A)
                tensorC = torchvision.ops.box_iou(tensorA,tensorB)
                numbOjectinJson = torch.argmax(tensorC).item()
                numinList = torch.argmax(tensorC).item()
                #print(numinList)
                cx = (datos["frames"][count-1]["annos"]["boxes_3d"][numinList][0])*0.3739
                cy = (datos["frames"][count-1]["annos"]["boxes_3d"][numinList][1])*0.3739
                cz = (datos["frames"][count-1]["annos"]["boxes_3d"][numinList][2])*0.3739
                #print("Bounding box ANNOS " + str(A[numinList][0]),str(A[numinList][1]),str(A[numinList][2]),str(A[numinList][3]))
                #print("Bounding box YOLO " + str(finalx1),str(finaly1),str(finalx2),str(finaly2) + "\n")
                realDistance = math.sqrt((cx)**2 + (cy)**2 + (cz)**2)
                distanceToCenter = (((finalx1+finalx2)/2)-960)
                realPosition.append(realDistance)
                estimated_distance.append(estimatedDistance)
                positionError.append(realDistance - estimatedDistance)
                distance_to_center.append(distanceToCenter)
                if(tipoVehiculo == "Coche"):
                    distancia_real_coche.append(realDistance)
                    distancia_estimada_coche.append(estimatedDistance)
                elif(tipoVehiculo == "Furgoneta"):
                    distancia_real_furgoneta.append(realDistance)
                    distancia_estimada_furgoneta.append(estimatedDistance)
                else:
                    distancia_real_camion.append(realDistance)
                    distancia_estimada_camion.append(estimatedDistance)
                if(realDistance < 5):
                    realPositionMin.append(realDistance)
                    positionErrorMin.append(realDistance - estimatedDistance)
                elif (realDistance > 15):
                    realPositionMax.append(realDistance)
                    positionErrorMax.append(realDistance-estimatedDistance)
                else:
                    realPositionMed.append(realDistance)
                    positionErrorMed.append(realDistance-estimatedDistance)

                trueAnnos = True

    realDistanceObject = math.sqrt((cx)**2 + (cy)**2 + (cz)**2)
    return realDistanceObject,trueAnnos

def mostrar_graficas():
    global error_velocity
    global difVel
    global difVelReal
    global positionError
    global realPosition
    global positionErrorMin
    global realPositionMin
    global positionErrorMed
    global realPositionMed
    global positionErrorMax
    global realPositionMax
    global distancia_real_coche
    global distancia_estimada_coche
    global distancia_real_furgoneta
    global distancia_estimada_furgoneta
    global distancia_real_camion
    global distancia_estimada_camion


    plt.figure(1)
    plt.scatter(realPositionMin,positionErrorMin)
    plt.xlabel('Distancia real del vehículo')
    plt.ylabel('Diferencia entre distancia estimada y distancia real')
    plt.title('Grafica 1')
    plt.savefig(ruta_graficas)

    plt.figure(2)
    plt.scatter(realPositionMed,positionErrorMed)
    plt.xlabel('Distancia real del vehículo')
    plt.ylabel('Diferencia entre distancia estimada y distancia real')
    plt.title('Grafica 2')
    plt.savefig(ruta_graficas)

    plt.figure(3)
    plt.scatter(realPositionMax,positionErrorMax)
    plt.xlabel('Distancia real del vehículo')
    plt.ylabel('Diferencia entre distancia estimada y distancia real')
    plt.title('Grafica 3')
    plt.savefig(ruta_graficas)

    plt.figure(4)
    plt.scatter(realPosition,positionError)
    plt.xlabel('Distancia real del vehículo')
    plt.ylabel('Diferencia entre distancia estimada y distancia real')
    plt.title('Grafica 4')
    plt.savefig(ruta_graficas)

    plt.figure(5)
    plt.scatter(realPosition,error_velocity)
    plt.xlabel('Distancia real del vehículo')
    plt.ylabel('Diferencia entre velocidad real y estimada')
    plt.title('Grafica 5')
    plt.savefig(ruta_graficas)

    plt.figure(5)
    plt.scatter(realPosition,error_velocity)
    plt.xlabel('Distancia real del vehículo')
    plt.ylabel('Diferencia entre velocidad real y estimada')
    plt.title('Grafica 5')
    plt.savefig(ruta_graficas)

    plt.figure(6)
    plt.scatter(distance_to_center,error_velocity)
    plt.xlabel('Diferencia entre centro h. vehículo y centro h. fotograma')
    plt.ylabel('Diferencia entre velocidad real y estimada')
    plt.title('Grafica 6')
    plt.savefig(ruta_graficas)

    plt.figure(7)
    plt.scatter(distance_to_center,positionError)
    plt.xlabel('Diferencia entre centro h. vehículo y centro h. fotograma')
    plt.ylabel('Diferencia entre distancia estimada y distancia real')
    plt.title('Grafica 7')
    plt.savefig(ruta_graficas)

    # Crear el histograma
    plt.figure(figsize=(10, 6))
    plt.hist(distancia_estimada_camion, alpha=0.5, label='Distancias Estimadas', edgecolor='black')
    plt.hist(distancia_real_camion, alpha=0.5, label='Distancias Reales', edgecolor='black')

    # Añadir título y etiquetas
    plt.title('Comparación de Distancias Estimadas vs. Distancias Reales')
    plt.xlabel('Distancia (m)')
    plt.ylabel('Frecuencia')

    # Añadir leyenda
    plt.legend(loc='upper right')

    # Crear el histograma
    plt.figure(figsize=(10, 6))
    plt.hist(distancia_estimada_furgoneta, alpha=0.5, label='Distancias Estimadas', edgecolor='black')
    plt.hist(distancia_real_furgoneta, alpha=0.5, label='Distancias Reales', edgecolor='black')

    # Añadir título y etiquetas
    plt.title('Comparación de Distancias Estimadas vs. Distancias Reales')
    plt.xlabel('Distancia (m)')
    plt.ylabel('Frecuencia')

    # Añadir leyenda
    plt.legend(loc='upper right')

    plt.show()


    distancia_real_coche_np = np.array(distancia_real_coche)
    distancia_estimada_coche_np = np.array(distancia_estimada_coche)
    distancia_real_furgoneta_np = np.array(distancia_real_furgoneta)
    distancia_estimada_furgoneta_np = np.array(distancia_estimada_furgoneta)
    distancia_real_camion_np = np.array(distancia_real_camion)
    distancia_estimada_camion_np = np.array(distancia_estimada_camion)
    # Calculando el error entre las distancias reales y estimadas
    erroresCoche = distancia_estimada_coche_np - distancia_real_coche_np
    erroresFurgoneta = distancia_estimada_furgoneta_np - distancia_real_furgoneta_np
    erroresCamion = distancia_estimada_camion_np - distancia_real_camion_np

    # Creando una matriz 2D (por ejemplo, podrías tener diferentes categorías en el eje Y si es útil)
    # En este caso simple, todas las distancias y errores se tratan de la misma manera
    matriz_de_errores_coche = np.array([erroresCoche])
    matriz_de_errores_camion = np.array([erroresCamion])
    matriz_de_errores_furgoneta = np.array([erroresFurgoneta])

    # Creando el mapa de calor
    plt.figure(8)
    plt.imshow(matriz_de_errores_coche, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Error en la estimación de la distancia')
    plt.title('Mapa de Calor de Errores en la Estimación de la Distancia')
    plt.xlabel('Índice de la muestra')
    plt.ylabel('Coches')
    plt.savefig(ruta_graficas)
    plt.show()

    plt.figure(9)
    plt.imshow(matriz_de_errores_furgoneta, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Error en la estimación de la distancia')
    plt.title('Mapa de Calor de Errores en la Estimación de la Distancia')
    plt.xlabel('Índice de la muestra')
    plt.ylabel('Furgonetas')
    plt.savefig(ruta_graficas)
    plt.show()

    plt.figure(10)
    plt.imshow(matriz_de_errores_camion, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Error en la estimación de la distancia')
    plt.title('Mapa de Calor de Errores en la Estimación de la Distancia')
    plt.xlabel('Índice de la muestra')
    plt.ylabel('Camiones')
    plt.savefig(ruta_graficas)
    plt.show()


video = os.path.join(os.getcwd(), 'video21.mp4')
analyze_video (video)
#realDistance(1081.4146, 550.4589, 1335.9025, 636.07294,str(1616451420300))