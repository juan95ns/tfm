# Ejecución:
# python face_recognition_video.py --encodings encodings/encodings.pickle --input input_videos/video.mp4 --output output_videos/video_output.avi

import time
from cv2 import cv2
import pickle
import imutils
import argparse
import face_recognition


def run():
    # Argumentos para la ejecución del programa
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--encodings", required=True, help="path encodings")
    argparser.add_argument("--input", required=True, help="path input video")
    argparser.add_argument("--output", required=True, help="path output video")
    args = vars(argparser.parse_args())

    # Cargamos los encodings de las caras
    print("[Paso 1] Cargando datos")
    pickle_encodings = pickle.loads(open(args["encodings"], "rb").read())

    # Cargamos el video en memoria
    video = cv2.VideoCapture(args["input"])
    writer = None

    process_video(args, pickle_encodings, video, writer)


def process_video(args, pickle_encodings, video, writer):
    iteration = 0
    # Bucle mientras haya frames en el video
    while True:
        # Obtener el siguiente frame del video y si no quedan, salimos del bucle
        (grabbed, frame) = video.read()

        if not grabbed:
            break

        # Convertimos el video a RGB y 750 de anchura para acelerar el proceso
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(frame, width=750)
        r = frame.shape[1] / float(rgb.shape[1])

        # Detectar las caras en el video
        print("[Paso 2] Detectando caras en frame {}".format(iteration))
        faces = face_recognition.face_locations(rgb, model="cnn")
        encodings = face_recognition.face_encodings(rgb, faces)
        names = list()

        # Recorremos los encodings de cada una de las caras detectadas
        print("[Paso 3] Identificando caras")
        for encoding in encodings:
            # Creamos un array de booleans con las comparaciones
            comparations = face_recognition.compare_faces(pickle_encodings["encodings"], encoding)
            name = "Desconocido"

            if True in comparations:
                # Guardamos el indice del array si la comparación es positiva
                comparations_index = [i for (i, b) in enumerate(comparations) if b == True]
                comparations_count = dict()

                # Contar las veces que sale la comparación por persona
                for i in comparations_index:
                    name = pickle_encodings["names"][i]
                    comparations_count[name] = comparations_count.get(name, 0) + 1

                # Obtener que persona obtiene el mayor resultado
                name = max(comparations_count, key=comparations_count.get)

            names.append(name)

        # Recorremos las caras que han sido reconocidas
        print("[Paso 4] Escribiendo caras")
        for ((top, right, bottom, left), name) in zip(faces, names):
            # Reescalamos las coordenadas en base al resto anterior
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)

            # Dibujamos la cara y el nombre
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Si el writer está vacio, lo inicializamos
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 24, (frame.shape[1], frame.shape[0]), True)

        if writer is not None:
            writer.write(frame)

        iteration = iteration + 1

    # Liberamos video y writer de memoria
    print("[Paso 5] Resultado final guardado")
    video.release()
    if writer is not None:
        writer.release()


if __name__ == "__main__":
    run()
