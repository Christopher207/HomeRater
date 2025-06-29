# map_downloader.py
import os
import folium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image
import io
import time
import re # Para expresiones regulares, útil para extraer el número del nombre del archivo

# --- Configuración del Chromedriver (¡AJUSTA ESTO!) ---
# Si chromedriver no está en tu PATH, descomenta y ajusta la siguiente línea:
# CHROMEDRIVER_PATH = '/path/to/your/chromedriver' # Ejemplo: '/usr/local/bin/chromedriver'

# O puedes dejarlo así si el chromedriver está en tu PATH o en el mismo directorio.
CHROMEDRIVER_SERVICE = None
try:
    # Intenta inicializar el servicio de Chromedriver asumiendo que está en PATH
    CHROMEDRIVER_SERVICE = ChromeService()
    # Una prueba rápida para ver si se puede encontrar el binario
    webdriver.Chrome(service=CHROMEDRIVER_SERVICE, options=webdriver.ChromeOptions()).quit()
    print("Chromedriver encontrado en el PATH o en el directorio actual.")
except Exception as e:
    print(f"Advertencia: Chromedriver no encontrado en el PATH o error de inicialización: {e}")
    print("Por favor, asegúrate de que Chromedriver esté instalado y accesible en tu PATH,")
    print("o descomenta y ajusta la variable CHROMEDRIVER_PATH en map_downloader.py.")
    # Si sabes la ruta específica, puedes inicializarlo así:
    # CHROMEDRIVER_SERVICE = ChromeService(executable_path=CHROMEDRIVER_PATH)


def get_next_image_index(image_folder_path: str) -> int:
    """
    Determina el siguiente índice numérico disponible para una nueva imagen
    leyendo los nombres de los archivos existentes en la carpeta.
    Asume que los nombres de archivo son números enteros (ej., '1.png', '10.jpg').

    Args:
        image_folder_path (str): Ruta a la carpeta que contiene las imágenes.

    Returns:
        int: El siguiente índice numérico. Si no hay imágenes, retorna 1.
    """
    max_index = 0
    if os.path.exists(image_folder_path):
        for filename in os.listdir(image_folder_path):
            # Extraer el número del nombre del archivo (ej. "123.png" -> 123)
            match = re.match(r'(\d+)\.(png|jpg|jpeg)', filename.lower())
            if match:
                try:
                    index = int(match.group(1))
                    if index > max_index:
                        max_index = index
                except ValueError:
                    # Ignorar archivos que no tengan un número válido como nombre
                    pass
    return max_index + 1

def save_map_image_for_coordinates(lat: float, lon: float, output_directory: str = "MapasDescargados",
                                   radius_meters: int = 200, image_quality: int = 85,
                                   image_size: tuple = (800, 800)) -> str or None:
    """
    Genera y guarda una imagen de mapa para unas coordenadas dadas,
    asignándole un índice incrementado.

    Args:
        lat (float): Latitud del punto central.
        lon (float): Longitud del punto central.
        output_directory (str): Directorio donde se guardará la imagen.
        radius_meters (int): Radio del círculo en metros.
        image_quality (int): Calidad de la imagen (para JPG, 0-100).
        image_size (tuple): Tamaño de la imagen (ancho, alto) en píxeles.

    Returns:
        str: La ruta completa del archivo de imagen guardado, o None si falla.
    """
    os.makedirs(output_directory, exist_ok=True)

    # Calcular el siguiente índice
    next_index = get_next_image_index(output_directory)
    output_filename = f"{next_index}.png" # Se recomienda PNG para mapas Folium por la calidad
    output_filepath = os.path.join(output_directory, output_filename)

    # Calcular el zoom aproximado (ajustado para una vista más cercana del radio)
    # Un zoom de 15 es un buen punto de partida para ver un radio de 200m
    zoom = 50

    # Crear un mapa centrado en las coordenadas
    m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles="OpenStreetMap")

    # Añadir un marcador y un círculo
    folium.Marker([lat, lon], tooltip="Punto central").add_to(m)


    # Guardar el mapa como HTML temporal
    temp_html = os.path.join(output_directory, f"temp_map_{next_index}.html")
    m.save(temp_html)

    # Usar Selenium para capturar la imagen
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Ejecutar en modo sin cabeza
    options.add_argument(f"--window-size={image_size[0]},{image_size[1]}")  # Usar el tamaño especificado
    options.add_argument("--hide-scrollbars")  # Ocultar scrollbars
    options.add_argument("--disable-gpu")  # Añadido para mejor compatibilidad
    options.add_argument("--no-sandbox") # Necesario en algunos entornos (ej. Docker)
    options.add_argument("--disable-dev-shm-usage") # Para entornos con poca RAM (/dev/shm)

    driver = None
    try:
        if CHROMEDRIVER_SERVICE:
            driver = webdriver.Chrome(service=CHROMEDRIVER_SERVICE, options=options)
        else: # Si CHROMEDRIVER_SERVICE es None, intentar con el constructor predeterminado
            driver = webdriver.Chrome(options=options)

        driver.get(f"file://{os.path.abspath(temp_html)}")

        # Esperar a que el mapa cargue completamente.
        # Puedes mejorar esto esperando un elemento específico del mapa de Folium si es posible.
        # Por ahora, un tiempo fijo puede ser suficiente.
        time.sleep(3) # Aumentado el tiempo para mayor fiabilidad

        # Capturar la imagen
        screenshot = driver.get_screenshot_as_png()

        # Procesar y guardar la imagen con Pillow
        img = Image.open(io.BytesIO(screenshot))

        # Convertir a RGB si es RGBA, necesario para JPG. PNG puede manejar RGBA.
        if output_filepath.lower().endswith((".jpg", ".jpeg")) and img.mode == 'RGBA':
            img = img.convert('RGB')
        elif output_filepath.lower().endswith(".png") and img.mode == 'RGB':
            img = img.convert('RGBA') # Opcional: convertir a RGBA para mejor transparencia en PNG

        if output_filepath.lower().endswith((".jpg", ".jpeg")):
            img.save(output_filepath, quality=image_quality, optimize=True)
        elif output_filepath.lower().endswith(".png"):
            img.save(output_filepath, optimize=True, compress_level=9)
        else:
            img.save(output_filepath) # Guarda con la configuración predeterminada

        print(f"Imagen guardada como {output_filepath}")
        return output_filepath

    except Exception as e:
        print(f"Error al capturar o guardar la imagen para ({lat}, {lon}): {e}")
        return None
    finally:
        if driver:
            driver.quit()
        if os.path.exists(temp_html):
            os.remove(temp_html) # Eliminar el archivo HTML temporal

# Si este script se ejecuta directamente (para pruebas), lo hará
if __name__ == "__main__":
    TEST_MAPS_FOLDER = "MapasDescargados"
    os.makedirs(TEST_MAPS_FOLDER, exist_ok=True)

    # Crea algunas imágenes dummy para probar el incremento del índice
    print(f"Creando imágenes de prueba en {TEST_MAPS_FOLDER} para verificar el índice...")
    for i in [1, 5, 10]: # Crea algunos archivos con diferentes índices
        dummy_map_file = os.path.join(TEST_MAPS_FOLDER, f"{i}.png")
        # Crea un archivo vacío o un dummy muy pequeño para simular una imagen
        with open(dummy_map_file, 'w') as f:
            f.write("dummy")
        print(f"  - Creado: {dummy_map_file}")
    print("Imágenes de prueba creadas.\n")


    print("--- Ejecutando prueba de descarga de mapa para coordenadas específicas ---")
    test_lat = -12.046374 # Latitud de Lima, Perú
    test_lon = -77.042793 # Longitud de Lima, Perú

    saved_path = save_map_image_for_coordinates(test_lat, test_lon, output_directory=TEST_MAPS_FOLDER)

    if saved_path:
        print(f"\nImagen de mapa de prueba guardada en: {saved_path}")
    else:
        print("\nNo se pudo guardar la imagen de mapa de prueba.")

    # Limpia las imágenes de prueba
    print("\nLimpiando imágenes de prueba...")
    for filename in os.listdir(TEST_MAPS_FOLDER):
        if re.match(r'(\d+)\.(png|jpg|jpeg)', filename.lower()) or filename.startswith("temp_map_"):
            os.remove(os.path.join(TEST_MAPS_FOLDER, filename))
            print(f"  - Eliminado: {filename}")
    if not os.listdir(TEST_MAPS_FOLDER): # Si la carpeta está vacía, elimínala
        os.rmdir(TEST_MAPS_FOLDER)
    print("Prueba completada y limpieza realizada.")