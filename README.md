# CardioScan ❤️

Sistema inteligente para la predicción de supervivencia post-infarto basado en indicadores de ecocardiografía clínica.

## 🚀 Descripción
CardioScan es una aplicación desarrollada con **Streamlit** que utiliza modelos de Machine Learning (Redes Neuronales y Regresión Logística) para determinar la probabilidad de supervivencia de un paciente después de un ataque cardíaco, utilizando el dataset de Ecocardiogramas de UCI.

## ✨ Características
- **Análisis Individual**: Ingresa parámetros clínicos manualmente para obtener una predicción instantánea.
- **Procesamiento por Lotes**: Sube un archivo CSV con múltiples registros para análisis masivo.
- **Modelos Comparativos**: Compara el rendimiento entre una **Red Neuronal (MLP)** y una **Regresión Logística**.
- **Visualización**: Incluye matrices de confusión y métricas detalladas (Accuracy, F1-Score, Precision, Recall).
- **Interfaz Premium**: Diseño oscuro moderno con tipografía Syne y DM Sans.

## 🛠️ Tecnologías
- Python 3.x
- Streamlit
- Scikit-learn
- Pandas / Numpy
- Matplotlib / Seaborn

## 📦 Instalación
1. Clona el repositorio:
   ```bash
   git clone https://github.com/monteseliana91/CardioScan.git
   ```
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecuta la aplicación:
   ```bash
   streamlit run app.py
   ```

## 📊 Dataset
El proyecto utiliza el `echocardiogram.data` que contiene mediciones de ecocardiografía de pacientes que sufrieron ataques cardíacos. Algunas de las variables clave incluyen:
- Edad (Age)
- Derrame pericárdico (Pericardial Effusion)
- Acortamiento fraccional (Fractional Shortening)
- EPSS
- LVDD
- Índice de movimiento de la pared (Wall Motion Index)

---
Desarrollado para el análisis de supervivencia cardíaca.
