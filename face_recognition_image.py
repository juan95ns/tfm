# Ejecución:
# python face_recognition_image.py --encodings encodings/encodings.pickle --image input_images/ex1.png 

# Imports
from cv2 import cv2
import pickle
import argparse
import face_recognition


def run():
    # Argumentos para la ejecución del programa
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--encodings", required=True, help="path encodings")
    argparser.add_argument("--image", required=True, help="path input image")
    args = vars(argparser.parse_args())

    # Cargar los datos necesarios para la ejecución del programa y convertir la imagen de entrada a RGB
    print("[Paso 1] Cargando datos")
    pickle_encodings = pickle.loads(open(args["encodings"], "rb").read())
    image = cv2.imread(args["image"])
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detect_faces(args, pickle_encodings, image, rgb)


def detect_faces(args, pickle_encodings, image, rgb):
    names = list()

    # Detectar las coordenadas de la imagen que corresponden con caras
    print("[Paso 2] Procesando imagen de entrada")
    faces = face_recognition.face_locations(rgb, model="cnn")
    image_encodings = face_recognition.face_encodings(rgb, faces)

    # Recorremos los encodings de cada una de las caras detectadas
    print("[Paso 3] Detectando caras")
    for image_encoding in image_encodings:
        # Creamos un array de booleans con las comparaciones
        comparations = face_recognition.compare_faces(pickle_encodings["encodings"], image_encoding)
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

    print("[Paso 4] Resultado final")
    write_faces(image, faces, names)


def write_faces(image, faces, names):
    # Dibujar el cuadrado de la cara con el nombre
    for((top, right, bottom, left), name) in zip(faces, names):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("Image", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    run()