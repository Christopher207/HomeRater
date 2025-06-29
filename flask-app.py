# web app/flask-app.py
from flask import Flask, redirect, url_for, request, render_template, jsonify
import json
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import folium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from PIL import Image
import io
import time
import re
import joblib
from sklearn.preprocessing import StandardScaler

# Intenta importar ChromeDriverManager para una gestión más sencilla del Chromedriver
try:
    from webdriver_manager.chrome import ChromeDriverManager
    WEBDRIVER_MANAGER_AVAILABLE = True
except ImportError:
    WEBDRIVER_MANAGER_AVAILABLE = False
    print("Advertencia: webdriver_manager no está instalado. Instálalo con 'pip install webdriver-manager' para una gestión más fácil de Chromedriver.")


app = Flask(__name__)

# --- Configuración de Carpetas ---
MAPS_STORAGE_FOLDER = "MapasDescargados"
os.makedirs(MAPS_STORAGE_FOLDER, exist_ok=True)

# --- Configuración y Carga del Modelo VGG16 ---
vgg_model = None
TARGET_IMAGE_SIZE_VGG = (224, 224) # Tamaño esperado por VGG16

def load_vgg_model():
    global vgg_model
    if vgg_model is None:
        print("Cargando y calentando el modelo VGG16 por primera vez...")
        tf.keras.backend.clear_session()
        base_model = VGG16(weights='imagenet', include_top=True)
        vgg_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
        vgg_model.predict(np.zeros((1, *TARGET_IMAGE_SIZE_VGG, 3)), verbose=0)
        print("Modelo VGG16 cargado y listo.")
    return vgg_model

def load_image_for_vgg(path, target_size=None):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Advertencia: No se pudo cargar la imagen {path} para VGG.")
        return None
    if target_size:
        img = cv2.resize(img, target_size)
    return img

def get_last_image_vgg_features(image_folder_path: str) -> list or None:
    current_vgg_model = load_vgg_model()

    image_files = [f for f in os.listdir(image_folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"No se encontraron imágenes en la carpeta: {image_folder_path}")
        return None

    def get_image_index(filename):
        try:
            return int(os.path.splitext(filename)[0])
        except ValueError:
            return float('-inf')

    valid_image_files = [f for f in image_files if get_image_index(f) != float('-inf')]
    if not valid_image_files:
        print(f"No se encontraron imágenes con nombres de índice numéricos en la carpeta: {image_folder_path}")
        return None

    valid_image_files.sort(key=get_image_index)

    last_image_filename = valid_image_files[-1]
    last_image_path = os.path.join(image_folder_path, last_image_filename)

    print(f"Procesando la última imagen para VGG: {last_image_path}")

    img = load_image_for_vgg(last_image_path, target_size=TARGET_IMAGE_SIZE_VGG)
    if img is None:
        return None

    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    features = current_vgg_model.predict(img, verbose=0).flatten()
    return features[:5].tolist()

# --- Configuración y Funciones de Descarga de Mapas ---
CHROMEDRIVER_PATH = None

CHROMEDRIVER_SERVICE = None
if WEBDRIVER_MANAGER_AVAILABLE:
    try:
        CHROMEDRIVER_SERVICE = ChromeService(ChromeDriverManager().install())
        print("Chromedriver instalado y listo usando WebDriverManager.")
    except Exception as e:
        print(f"Advertencia: Falló la inicialización de Chromedriver con WebDriverManager: {e}")
        print("Intentando buscar Chromedriver en el PATH o en la ruta manual especificada.")
        WEBDRIVER_MANAGER_AVAILABLE = False
if not WEBDRIVER_MANAGER_AVAILABLE:
    if CHROMEDRIVER_PATH:
        try:
            CHROMEDRIVER_SERVICE = ChromeService(executable_path=CHROMEDRIVER_PATH)
            webdriver.Chrome(service=CHROMEDRIVER_SERVICE, options=webdriver.ChromeOptions()).quit()
            print(f"Chromedriver encontrado y listo en la ruta manual: {CHROMEDRIVER_PATH}")
        except Exception as e_manual:
            print(f"ERROR: Chromedriver NO encontrado en la ruta manual especificada: {CHROMEDRIVER_PATH}. Error: {e_manual}")
            print("Asegúrate de que Chromedriver esté instalado y accesible en tu PATH o ajusta CHROMEDRIVER_PATH.")
            CHROMEDRIVER_SERVICE = None
    else:
        try:
            CHROMEDRIVER_SERVICE = ChromeService()
            webdriver.Chrome(service=CHROMEDRIVER_SERVICE, options=webdriver.ChromeOptions()).quit()
            print("Chromedriver encontrado en el PATH o en el directorio actual.")
        except Exception as e_path:
            print(f"ERROR: Chromedriver NO encontrado en el PATH o error de inicialización: {e_path}")
            print("Por favor, instala Chromedriver o especifica su ruta en CHROMEDRIVER_PATH.")
            CHROMEDRIVER_SERVICE = None


def get_next_image_index_map(image_folder_path: str) -> int:
    max_index = 0
    if os.path.exists(image_folder_path):
        for filename in os.listdir(image_folder_path):
            match = re.match(r'(\d+)\.(png|jpg|jpeg)', filename.lower())
            if match:
                try:
                    index = int(match.group(1))
                    if index > max_index:
                        max_index = index
                except ValueError:
                    pass
    return max_index + 1

def save_map_image_for_coordinates(lat: float, lon: float, output_directory: str = MAPS_STORAGE_FOLDER,
                                   radius_meters: int = 200, image_quality: int = 85,
                                   image_size: tuple = (800, 800)) -> str or None:
    os.makedirs(output_directory, exist_ok=True)

    next_index = get_next_image_index_map(output_directory)
    output_filename = f"{next_index}.png"
    output_filepath = os.path.join(output_directory, output_filename)

    zoom = 15 # Zoom para un radio de 200m

    m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles="OpenStreetMap")
    folium.Marker([lat, lon], tooltip="Punto central").add_to(m)
    folium.Circle(
        location=[lat, lon],
        radius=radius_meters,
        color="#3186cc",
        fill=True,
        fill_color="#3186cc"
    ).add_to(m)

    temp_html = os.path.join(output_directory, f"temp_map_{next_index}.html")
    m.save(temp_html)

    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument(f"--window-size={image_size[0]},{image_size[1]}")
    options.add_argument("--hide-scrollbars")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = None
    try:
        if CHROMEDRIVER_SERVICE:
            driver = webdriver.Chrome(service=CHROMEDRIVER_SERVICE, options=options)
        else:
            driver = webdriver.Chrome(options=options)

        driver.get(f"file://{os.path.abspath(temp_html)}")
        time.sleep(3) # Esperar a que el mapa cargue completamente

        screenshot = driver.get_screenshot_as_png()
        img = Image.open(io.BytesIO(screenshot))

        if output_filepath.lower().endswith((".jpg", ".jpeg")) and img.mode == 'RGBA':
            img = img.convert('RGB')
        elif output_filepath.lower().endswith(".png") and img.mode == 'RGB':
            img = img.convert('RGBA')

        if output_filepath.lower().endswith((".jpg", ".jpeg")):
            img.save(output_filepath, quality=image_quality, optimize=True)
        elif output_filepath.lower().endswith(".png"):
            img.save(output_filepath, optimize=True, compress_level=9)
        else:
            img.save(output_filepath)

        print(f"Imagen de mapa guardada como {output_filepath}")
        return output_filepath

    except Exception as e:
        print(f"Error al capturar o guardar la imagen para ({lat}, {lon}): {e}")
        return None
    finally:
        if driver:
            driver.quit()
        if os.path.exists(temp_html):
            os.remove(temp_html)

# --- Carga del Modelo de Regresión homeRater.joblib ---
MODEL_PATH = 'homeRater.joblib'
home_rater_model = None

try:
    home_rater_model = joblib.load(MODEL_PATH)
    print(f"Modelo de regresión '{MODEL_PATH}' cargado correctamente.")
except Exception as e:
    print(f"ERROR: No se pudo cargar el modelo de regresión '{MODEL_PATH}'. Asegúrate de que el archivo exista y esté accesible. Error: {e}")

# --- Carga del Modelo K-Means, su Scaler y los Precios Promedio por M2 ---
KMEANS_MODEL_PATH = 'kmeans_model.joblib'
KMEANS_SCALER_PATH = 'kmeans_scaler.joblib' # Ruta para el StandardScaler
KMEANS_AVG_PRICES_PATH = 'kmeans_avg_prices.json'

kmeans_model = None
kmeans_avg_prices_per_m2 = None # Renombramos para mayor claridad, aunque la variable final será 'kmeans_cluster_avg_price'
kmeans_scaler = None

try:
    kmeans_model = joblib.load(KMEANS_MODEL_PATH)
    print(f"Modelo K-Means cargado desde: {KMEANS_MODEL_PATH}")
except Exception as e:
    print(f"ERROR: No se pudo cargar el modelo K-Means desde '{KMEANS_MODEL_PATH}'. Error: {e}")

try:
    kmeans_scaler = joblib.load(KMEANS_SCALER_PATH)
    print(f"Scaler de K-Means cargado desde: {KMEANS_SCALER_PATH}")
except Exception as e:
    print(f"ERROR: No se pudo cargar el scaler de K-Means desde '{KMEANS_SCALER_PATH}'. Error: {e}")

try:
    with open(KMEANS_AVG_PRICES_PATH, 'r') as f:
        kmeans_avg_prices_per_m2 = json.load(f)
        # Convertir claves de string a int si es necesario
        kmeans_avg_prices_per_m2 = {int(k): v for k, v in kmeans_avg_prices_per_m2.items()}
    print(f"Precios promedio por m2 de K-Means cargados desde: {KMEANS_AVG_PRICES_PATH}")
except Exception as e:
    print(f"ERROR: No se pudieron cargar los precios promedio por m2 de K-Means desde '{KMEANS_AVG_PRICES_PATH}'. Error: {e}")


# --- Carga del Modelo VGG16 al iniciar la aplicación Flask ---
with app.app_context():
    load_vgg_model()
    print(f"La aplicación Flask ha cargado el modelo VGG16 y la carpeta de mapas es: {MAPS_STORAGE_FOLDER}")


# --- Rutas de la aplicación Flask ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/map")
def map_page():
    return render_template("map.html")

@app.route("/tasarInmueble")
def tasar_inmueble_page():
    return render_template("tasarInmueble.html")

@app.route('/api/properties')
def get_properties():
    properties_path = os.path.join(app.root_path, 'static', 'data', 'properties.json')
    try:
        with open(properties_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({"error": f"Archivo no encontrado: {properties_path}"}), 404
    except json.JSONDecodeError:
        return jsonify({"error": f"Error al decodificar JSON en: {properties_path}"}), 500

@app.route('/success/<name>')
def success(name):
    return 'welcome %s' % name

@app.route('/test2')
def test():
    return render_template("test2.html")

@app.route('/basic')
def basic():
    return render_template("basic.html")

@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        user = request.form['nm']
        return redirect(url_for('success', name=user))
    else:
        user = request.args.get('nm')
        return redirect(url_for('success', name=user))

@app.route('/api/extract_vgg_features', methods=['GET'])
def extract_vgg_features_endpoint():
    try:
        features = get_last_image_vgg_features(MAPS_STORAGE_FOLDER)

        if features is not None:
            return jsonify({
                "status": "success",
                "message": "Características VGG extraídas correctamente.",
                "features": features
            })
        else:
            return jsonify({
                "status": "error",
                "message": "No se pudieron extraer las características VGG. Verifique la carpeta de imágenes."
            }), 500
    except Exception as e:
        app.logger.error(f"Error al procesar la solicitud de extracción VGG: {e}")
        return jsonify({
            "status": "error",
            "message": f"Ocurrió un error interno: {str(e)}"
        }), 500

@app.route('/api/download_map', methods=['POST'])
def download_map_endpoint():
    data = request.get_json()

    if not data:
        return jsonify({
            "status": "error",
            "message": "Se espera un JSON con 'latitude' y 'longitude'."
        }), 400

    lat = data.get('latitude')
    lon = data.get('longitude')

    if lat is None or lon is None:
        return jsonify({
            "status": "error",
            "message": "Faltan 'latitude' o 'longitude' en el JSON."
        }), 400

    try:
        lat = float(lat)
        lon = float(lon)
    except ValueError:
        return jsonify({
            "status": "error",
            "message": "La latitud y longitud deben ser valores numéricos."
        }), 400

    print(f"Recibida solicitud para descargar mapa en Lat: {lat}, Lon: {lon}")
    try:
        saved_file_path = save_map_image_for_coordinates(lat, lon, output_directory=MAPS_STORAGE_FOLDER)

        if saved_file_path:
            return jsonify({
                "status": "success",
                "message": "Imagen de mapa generada y guardada correctamente.",
                "file_path": saved_file_path
            })
        else:
            return jsonify({
                "status": "error",
                "message": "No se pudo generar o guardar la imagen del mapa."
            }), 500
    except Exception as e:
        app.logger.error(f"Error al procesar la solicitud de descarga de mapa: {e}")
        return jsonify({
            "status": "error",
            "message": f"Ocurrió un error interno: {str(e)}"
        }), 500

@app.route('/predict_price', methods=['POST'])
def predict_price():
    if home_rater_model is None:
        return jsonify({'error': 'El modelo de predicción de precios no está cargado. Por favor, contacta al administrador.'}), 500
    if kmeans_model is None or kmeans_avg_prices_per_m2 is None or kmeans_scaler is None:
        return jsonify({'error': 'El modelo K-Means, su scaler o los precios promedio por m2 de los clusters no están cargados. Contacta al administrador.'}), 500

    try:
        data = request.get_json()
        latitud = data.get('latitud')
        longitud = data.get('longitud')
        area_m2 = data.get('area_m2')

        if any(v is None for v in [latitud, longitud, area_m2]):
            return jsonify({'error': 'Datos incompletos. Se requieren latitud, longitud y area_m2.'}), 400

        try:
            latitud = float(latitud)
            longitud = float(longitud)
            area_m2 = int(area_m2)
            if area_m2 <= 0:
                raise ValueError("El área debe ser un número entero positivo.")
        except ValueError as e:
            return jsonify({'error': f'Error en el formato de los datos: {e}'}), 400

        print(f"Recibidos datos para predicción: Lat={latitud}, Lng={longitud}, Area={area_m2} m²")

        # 1. Capturar la imagen del mapa con la función existente
        map_image_filepath = save_map_image_for_coordinates(latitud, longitud, radius_meters=200, image_size=(800, 800))

        if not map_image_filepath:
            return jsonify({'error': 'No se pudo capturar la imagen del mapa para la predicción.'}), 500

        # 2. Extraer características VGG de la imagen capturada
        vgg_features = get_last_image_vgg_features(MAPS_STORAGE_FOLDER)

        if vgg_features is None:
            if os.path.exists(map_image_filepath):
                os.remove(map_image_filepath)
                print(f"Eliminado archivo de imagen: {map_image_filepath} debido a falla en extracción VGG.")
            return jsonify({'error': 'No se pudieron extraer las características VGG de la imagen del mapa.'}), 500

        print(f"Características VGG extraídas: {vgg_features}")

        # --- 3. Calcular kmeans_cluster_avg_price (ahora el precio promedio por m2 del cluster) ---
        # Prepara las coordenadas para el K-Means model y ESCALALAS
        coords_for_kmeans = np.array([[latitud, longitud]])
        coords_for_kmeans_scaled = kmeans_scaler.transform(coords_for_kmeans)

        # Asignar el cluster a las nuevas coordenadas
        cluster_id = kmeans_model.predict(coords_for_kmeans_scaled)[0]

        # Obtener el precio promedio POR M2 del cluster
        avg_price_per_m2_of_cluster = kmeans_avg_prices_per_m2.get(cluster_id, 0.0)

        # Ahora, para la característica 'kmeans_cluster_avg_price' que el modelo espera,
        # multiplicamos el precio promedio por m2 del cluster por el área de la propiedad actual.
        # Esto es lo que el modelo fue entrenado para ver si usaste la variable original
        # 'kmeans_cluster_avg_price' en tu RandomForestRegressor (que internamente era precio total).
        # Si tu RandomForestRegressor fue entrenado con el 'precioSoles_por_m2' de los clusters,
        # entonces solo pasarías 'avg_price_per_m2_of_cluster'.
        # Dada tu instrucción de "no cambiar el nombre de la variable", asumimos que
        # tu modelo espera una característica que es un precio total estimado basado en el cluster.

        # Vamos a recalcular 'kmeans_cluster_avg_price' para que sea un precio total estimado
        # basado en el precio promedio por m2 del cluster y el área_m2 de la propiedad actual.
        kmeans_cluster_avg_price = avg_price_per_m2_of_cluster * area_m2

        print(f"Cluster K-Means asignado: {cluster_id}, Precio promedio por m2 del cluster: {avg_price_per_m2_of_cluster}, Característica 'kmeans_cluster_avg_price' calculada: {kmeans_cluster_avg_price}")


        # 4. Preparar todas las características para el modelo (AHORA SON 9)
        # Asegúrate de que el orden de las características sea el mismo que el modelo espera.
        # El orden debe ser: [latitud, longitud, area_m2, vgg_feat1, ..., vgg_feat5, kmeans_cluster_avg_price]
        features_for_model = np.array([[latitud, longitud, area_m2] + vgg_features + [kmeans_cluster_avg_price]])

        # 5. Realizar la predicción
        predicted_price = home_rater_model.predict(features_for_model)[0]

        # 6. Opcional: Eliminar la imagen capturada después de extraer las características
        if os.path.exists(map_image_filepath):
            os.remove(map_image_filepath)
            print(f"Eliminado archivo de imagen temporal: {map_image_filepath}")

        predicted_price = round(predicted_price, 2)

        return jsonify({'predicted_price': float(predicted_price)})

    except ValueError as ve:
        return jsonify({'error': f'Error en el formato de los datos: {ve}'}), 400
    except Exception as e:
        app.logger.error(f"Error inesperado durante la predicción: {e}", exc_info=True)
        return jsonify({'error': f'Ocurrió un error interno del servidor: {str(e)}'}), 500

if __name__ == '__main__':
    properties_json_path = os.path.join(app.root_path, 'static', 'data', 'properties.json')
    if not os.path.exists(properties_json_path):
        try:
            from preprocess import preprocess_txt_to_json
            preprocess_txt_to_json(os.path.join(app.root_path, 'depaAlqUrbaniaConsolidado.txt'), properties_json_path)
            print(f"Archivo 'properties.json' generado en: {properties_json_path}")
        except ImportError:
            print("Error: No se encontró el módulo 'preprocess'. Asegúrate de que 'preprocess.py' exista y sea importable.")
            print("No se pudo generar 'properties.json'. La API /api/properties podría fallar.")
        except FileNotFoundError:
            print("Error: 'depaAlqUrbaniaConsolidado.txt' no encontrado. Asegúrate de que esté en el directorio de la aplicación.")
            print("No se pudo generar 'properties.json'. La API /api/properties podría fallar.")
        except Exception as e:
            print(f"Error inesperado durante la preprocesamiento: {e}")
            print("No se pudo generar 'properties.json'. La API /api/properties podría fallar.")

    app.run(debug=True)