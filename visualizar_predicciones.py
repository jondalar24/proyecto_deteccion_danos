import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# =====================
# CONFIGURACIÓN INICIAL
# =====================
BATCH_SIZE = 32
IMG_ROWS, IMG_COLS = 224, 224
SEED = 42

# RUTAS
MODEL_PATH = "modelo_entrenado.h5"
DATASET_DIR = "aircraft_damage_dataset_v1/test"

# ==============
# CARGAR MODELO
# ==============
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el archivo del modelo: {MODEL_PATH}. Ejecuta primero main.py")

model = load_model(MODEL_PATH)
print(" Modelo cargado correctamente")

# ============================
# CARGAR GENERADOR DE TEST
# ============================
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_ROWS, IMG_COLS),
    batch_size=BATCH_SIZE,
    seed=SEED,
    class_mode='binary',
    shuffle=False
)

# ============================
# FUNCIONES DE VISUALIZACIÓN
# ============================
def plot_image_with_title(image, true_label, predicted_label, class_names):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    true_label_name = class_names[true_label]
    pred_label_name = class_names[predicted_label]
    plt.title(f"True: {true_label_name}\nPred: {pred_label_name}")
    plt.axis('off')
    plt.show()

def test_model_on_image(test_generator, model, index_to_plot=0):
    test_images, test_labels = next(test_generator)
    predictions = model.predict(test_images)
    predicted_classes = (predictions > 0.5).astype(int).flatten()

    class_indices = test_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}

    image_to_plot = test_images[index_to_plot]
    true_label = int(test_labels[index_to_plot])
    predicted_label = int(predicted_classes[index_to_plot])

    plot_image_with_title(image=image_to_plot,
                          true_label=true_label,
                          predicted_label=predicted_label,
                          class_names=class_names)

# ======================
# INTERFAZ INTERACTIVA
# ======================
if __name__ == "__main__":
    max_index = BATCH_SIZE - 1
    try:
        index = int(input(f"Elige un número de imagen entre 0 y {max_index}: "))
        if 0 <= index <= max_index:
            test_model_on_image(test_generator, model, index_to_plot=index)
        else:
            print(f" Índice fuera de rango. Debe estar entre 0 y {max_index}")
    except Exception as e:
        print(f" Error: {e}")
