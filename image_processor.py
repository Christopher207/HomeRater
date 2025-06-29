# image_processor.py
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

# --- Configuración global del modelo VGG16 ---
# Estas variables se inicializarán solo una vez cuando se importe el módulo.
vgg_model = None
TARGET_IMAGE_SIZE = (224, 224) # Tamaño esperado por VGG16

def load_vgg_model():
    """Carga y calienta el modelo VGG16. Se llamará una sola vez."""
    global vgg_model
    if vgg_model is None:
        print("Cargando y calentando el modelo VGG16 por primera vez...")
        # Limpiar la sesión de Keras para asegurar un estado limpio
        tf.keras.backend.clear_session()
        base_model = VGG16(weights='imagenet', include_top=True)
        # Extraemos las características de la capa 'fc2' (la última capa densa antes de la salida de clasificación)
        vgg_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
        # Warm-up para GPU (realiza una predicción dummy para precargar el modelo en la GPU)
        vgg_model.predict(np.zeros((1, *TARGET_IMAGE_SIZE, 3)), verbose=0)
        print("Modelo VGG16 cargado y listo.")
    return vgg_model

def load_image(path, target_size=None):
    """Carga y redimensiona una imagen eficientemente."""
    img = cv2.imread(path, cv2.IMREAD_COLOR) # Asegurarse de cargar en color para VGG
    if img is None:
        print(f"Advertencia: No se pudo cargar la imagen {path}")
        return None
    if target_size:
        img = cv2.resize(img, target_size)
    return img

def get_last_image_vgg_features(image_folder_path: str) -> list or None:
    """
    Extrae el vector de características VGG16 de la última imagen
    (con el índice más alto) de una carpeta y retorna los primeros 5 features.

    Args:
        image_folder_path (str): Ruta a la carpeta que contiene las imágenes.

    Returns:
        list: Una lista con los primeros 5 valores del vector de características VGG16,
              o None si no se encuentra ninguna imagen o hay un error.
    """
    # Asegurarse de que el modelo esté cargado
    current_vgg_model = load_vgg_model()

    # Obtener lista de archivos de imagen y ordenarlos por índice numérico
    image_files = [f for f in os.listdir(image_folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"No se encontraron imágenes en la carpeta: {image_folder_path}")
        return None

    # Ordenar los archivos por el índice numérico en su nombre (ej. '1.png', '10.jpg')
    def get_image_index(filename):
        try:
            return int(os.path.splitext(filename)[0])
        except ValueError:
            # Si el nombre no es numérico, lo tratamos como -1 para que quede al principio
            # o podrías optar por ignorarlo. Para este caso, lo excluimos de la consideración.
            return float('-inf') # Esto asegura que los no-numéricos no sean "últimos"

    # Filtrar solo archivos con nombres numéricos válidos para el ordenamiento
    valid_image_files = [f for f in image_files if get_image_index(f) != float('-inf')]
    if not valid_image_files:
        print(f"No se encontraron imágenes con nombres de índice numéricos en la carpeta: {image_folder_path}")
        return None

    valid_image_files.sort(key=get_image_index)

    # La última imagen será la que tenga el índice numérico más alto
    last_image_filename = valid_image_files[-1]
    last_image_path = os.path.join(image_folder_path, last_image_filename)

    print(f"Procesando la última imagen: {last_image_path}")

    # Cargar y preprocesar la imagen para VGG16
    img = load_image(last_image_path, target_size=TARGET_IMAGE_SIZE)
    if img is None:
        return None

    # VGG16 espera una imagen BGR que se convierte a RGB y se normaliza
    # (cv2.imread carga en BGR por defecto)
    img = preprocess_input(img)

    # Añadir una dimensión de batch (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)

    # Extraer características VGG
    features = current_vgg_model.predict(img, verbose=0).flatten()

    # Retornar los primeros 5 features como una lista
    return features[:5].tolist()

# Si este script se ejecuta directamente (para pruebas), lo hará
if __name__ == "__main__":
    # DIRECTORIO DE IMÁGENES DE PRUEBA
    TEST_IMAGE_FOLDER = "MapasDescargados"
    os.makedirs(TEST_IMAGE_FOLDER, exist_ok=True)

    # Crea algunas imágenes de ejemplo para simular la carpeta MapasDescargados
    # Elimina esto cuando uses tus propias imágenes reales en un entorno de producción
    print(f"Creando imágenes de ejemplo en {TEST_IMAGE_FOLDER}...")
    for i in range(1, 6):
        dummy_image_path = os.path.join(TEST_IMAGE_FOLDER, f"{i}.png")
        if not os.path.exists(dummy_image_path):
            dummy_image = np.zeros((*TARGET_IMAGE_SIZE, 3), dtype=np.uint8) + i * 20 # Crea una imagen de color diferente
            cv2.imwrite(dummy_image_path, dummy_image)
    # También añade una imagen con un índice alto para probar que se selecciona la última
    dummy_image_path_high_index = os.path.join(TEST_IMAGE_FOLDER, "100.png")
    if not os.path.exists(dummy_image_path_high_index):
        dummy_image_high = np.zeros((*TARGET_IMAGE_SIZE, 3), dtype=np.uint8) + 250
        cv2.imwrite(dummy_image_high_index, dummy_image_high)
    print("Imágenes de ejemplo creadas.")

    print("\n--- Ejecutando prueba de extracción de características VGG ---")
    first_5_vgg_features = get_last_image_vgg_features(TEST_IMAGE_FOLDER)

    if first_5_vgg_features is not None:
        print("\nPrimeros 5 features VGG de la última imagen:")
        print(first_5_vgg_features)
    else:
        print("No se pudieron extraer los features de VGG.")

    # Limpiar imágenes de ejemplo después de la prueba
    print(f"Eliminando imágenes de ejemplo de {TEST_IMAGE_FOLDER}...")
    for i in range(1, 6):
        dummy_image_path = os.path.join(TEST_IMAGE_FOLDER, f"{i}.png")
        if os.path.exists(dummy_image_path):
            os.remove(dummy_image_path)
    if os.path.exists(os.path.join(TEST_IMAGE_FOLDER, "100.png")):
        os.remove(os.path.join(TEST_IMAGE_FOLDER, "100.png"))
    # Solo elimina la carpeta si está vacía
    if os.path.exists(TEST_IMAGE_FOLDER) and not os.listdir(TEST_IMAGE_FOLDER):
        os.rmdir(TEST_IMAGE_FOLDER)
    print("Prueba completada y limpieza realizada.")