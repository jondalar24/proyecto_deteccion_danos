# ✈️ Detección de Daños en Aeronaves con Deep Learning

Este proyecto demuestra cómo utilizar redes neuronales convolvolucionales preentrenadas (CNN) para detectar **abolladuras (`dent`) o grietas (`crack`)** en el fuselaje de aviones, a partir de imágenes reales. Además, se incluye una función de **diagnóstico visual** para mostrar predicciones del modelo sobre nuevas imágenes.

---

## 🧠 Tecnologías utilizadas

- **Python 3.10**
- **TensorFlow / Keras** (modelado y entrenamiento)
- **VGG16** (modelo CNN preentrenado en ImageNet, usado como extractor de características)
- **Matplotlib** (visualización de métricas e imágenes)
- **Transformers (BLIP)** de HuggingFace *(previsto para futuras versiones)*

---

## 📁 Estructura esperada

La raíz del proyecto debe contener:

```bash
deteccion_de_danos/
├── aircraft_damage_dataset_v1.tar   # 📦 Dataset original
├── main.py                          # 🔁 Entrenamiento completo
├── visualizar_predicciones.py       # 🖼️ Visualización de predicciones
├── requirements.txt
├── .gitignore
└── ...
```

Y tras ejecutar `main.py`, se generará:

```bash
aircraft_damage_dataset_v1/
├── train/
│   ├── dent/
│   └── crack/
├── valid/
│   ├── dent/
│   └── crack/
└── test/
    ├── dent/
    └── crack/
```

---

## 🛠️ Instalación y entorno recomendado

Se recomienda trabajar en un entorno virtual:

```bash
# 1. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt
```

---

## 🚀 Cómo ejecutar el proyecto

### 1️⃣ Entrenamiento del modelo

```bash
python main.py
```

📌 Este script:

- Extrae el dataset automáticamente
- Preprocesa las imágenes con `ImageDataGenerator`
- Usa **VGG16** como extractor de características
- Añade capas densas personalizadas
- Entrena durante 5 épocas
- Guarda el modelo como `modelo_entrenado.h5`
- Muestra las curvas de pérdida y precisión

---

### 2️⃣ Visualizar predicciones del modelo

```bash
python visualizar_predicciones.py
```

🔍 Este script:

- Carga el modelo entrenado
- Solicita un índice (número entre 0 y N)
- Muestra una imagen del conjunto de test
- Indica la clase real y la predicha por el modelo

---

## ❗ Nota sobre el modelo `.h5`

El archivo `modelo_entrenado.h5` **no está en el repositorio** (supera los 100MB permitidos por GitHub).  
👉 **Se genera automáticamente** al ejecutar `main.py`.

---

## 📌 Conclusión

Este proyecto muestra:

✅ Cómo aplicar **Transfer Learning**  
✅ Un flujo completo con **Keras**  
✅ Visualización de métricas y predicciones  
✅ Buenas prácticas de estructura de proyecto reproducible

---



**Autor:** [@jondalar24](https://github.com/jondalar24)
