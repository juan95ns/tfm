# Ejecución:
# python face_encoding.py --dataset dataset --encodings encodings/encodings.pickle

# Imports
from imutils import paths
import os
from cv2 import cv2
import pickle
import argparse
import face_recognition


def run():
    # Argumentos para la ejecución del programa
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", required=True, help="path dataset")
    argparser.add_argument("--encodings", required=True, help="path output encodings")
    args = vars(argparser.parse_args())

    # Crear la lista de rutas de la imagen con formato:
    # dataset\\alfred_molina\\00000000.jpg
    image_paths = list(paths.list_images(args["dataset"]))
    print("[Paso 1] Rutas de las imagenes cargadas.")

    process_faces(args, image_paths)


def process_faces(args, image_paths):
    # Listas para guardar los datos de las imagenes
    encoding_list = list()
    name_list = list()

    image_count = len(image_paths)

    # Recorrer las imagenes una a una
    print("[Paso 2] Codificando caras.")
    for iteration, path in enumerate(image_paths):
        name = path.split(os.path.sep)[-2]

        # Convertir imagen a RGB
        image = cv2.imread(path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detectar las coordenadas que corresponden a la cara en la imagen
        face = face_recognition.face_locations(rgb, model="cnn")

        # Procesar la cara
        encodings = face_recognition.face_encodings(rgb, face)

        # Poner los vectores resultantes y los nombres en la lista definitiva
        for encoding in encodings:
            encoding_list.append(encoding)
            name_list.append(encoding)

        print("Imagen {}/{}.".format(iteration + 1, image_count))

    write_encodings(args, encoding_list, name_list)


def write_encodings(args, encoding_list, name_list):
    # Escribir los encodings
    print("[Paso 3] Escribiendo encodings")
    data = {"encodings": encoding_list, "names": name_list}
    f = open(args["encodings"], "wb")
    f.write(pickle.dumps(data))
    f.close()


if __name__ == "__main__":
    run()
