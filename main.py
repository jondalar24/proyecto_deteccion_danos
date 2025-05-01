"""
main.py — Proyecto personal: Clasificación y Descripción de Daños en Aviones

Este script realiza:
1. Extracción del dataset desde archivo local (formato .tar)
2. Preparación de carpetas (train / valid / test)
3. Normalización básica de datos de imagen
4. Entrenamiento y adaptación del modelo VGG16
5. Visualización de resultados.

Autor: Ángel Calvar
"""

import os
import random
import shutil
import tarfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 1. Configuración general
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 2. Semillas para reproducibilidad
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 3. Configuración del modelo
BATCH_SIZE = 32
EPOCHS = 5
IMG_ROWS, IMG_COLS = 224, 224
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 3)

# 4. Extracción del dataset
print("Extrayendo el dataset...\n")
project_root = Path(__file__).resolve().parent
dataset_tar = project_root / "aircraft_damage_dataset_v1.tar"
dataset_dir = project_root / "aircraft_damage_dataset_v1"

if not dataset_tar.exists():
    raise FileNotFoundError("Dataset .tar no encontrado en la raíz del proyecto")

if dataset_dir.exists():
    shutil.rmtree(dataset_dir)

with tarfile.open(dataset_tar, "r") as tar:
    tar.extractall(path=project_root)

# 5. Carga imágenenes y directorios
# 5.1 cargamos datos
print("Carga de datos...\n")
tain_datagen = ImageDataGenerator(rescale=1. / 255)
valid_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# 5.2 Creamos las carpetas
train_dir = dataset_dir / 'train'
valid_dir = dataset_dir / 'valid'
test_dir = dataset_dir / 'test'

# 5.3 Crea los generadores de entrenamiento, validación y test
print("Creando generadores de entrenamiento, validación y test...\n")
train_generator = tain_datagen.flow_from_directory(
    train_dir, target_size=(IMG_ROWS, IMG_COLS), batch_size=BATCH_SIZE,
    class_mode='binary', shuffle=True, seed=SEED)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir, target_size=(IMG_ROWS, IMG_COLS), batch_size=BATCH_SIZE,
    class_mode='binary', shuffle=False, seed=SEED)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(IMG_ROWS, IMG_COLS), batch_size=BATCH_SIZE,
    class_mode='binary', shuffle=False, seed=SEED)

# 6. Carga y adaptación del modelo VGG16
print("cargando el modelo VGG16...\n")
base_model = VGG16(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
output = base_model.layers[-1].output
output = Flatten()(output)
base_model = Model(base_model.input, output)
for layer in base_model.layers:
    layer.trainable = False

model = Sequential([
    base_model,
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 7. Entrenamiento
print("Entrenando...\n")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=valid_generator
)

# 8. Visualización de pérdida y precisión
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# 9. Evaluación final
loss, accuracy = model.evaluate(test_generator)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print("Modelo cargado y entrenado correctamente\n")

# Guardar modelo
model.save("modelo_entrenado.h5")
print("Modelo guardado como 'modelo_entrenado.h5'")
