# 🌫 Formulario Multivariable para Predicción de PM2.5

Este proyecto es una aplicación desarrollada con **Python** y **Streamlit** para predecir el valor de **PM2.5** a 24 horas, utilizando un enfoque de *stacking* con redes neuronales y modelos de machine learning.

Permite al usuario subir un archivo Excel con variables ambientales y obtener una predicción automática del contaminante PM2.5.

---

## 📂 Estructura del Proyecto

```
FORMULARIO_MULTIVARIABLE/
├── app.py                    # Aplicación principal en Streamlit
├── requirements.txt          # Dependencias Python
├── RNN_modelo_multivariable.h5  # Modelo RNN entrenado
├── stacking_model.pkl        # Modelo de stacking (ej. RandomForest, XGBoost)
├── scalers.pkl              # Scalers usados para normalización
```

---

## 🧠 Requisitos Previos

- **✔** Python 3.10.10 → [Descargar](https://www.python.org/downloads/windows/)

---

## ⚙️ Instalación y Ejecución

### 1. Crear entorno virtual (opcional pero recomendado)

```bash
python -m venv venv
venv\Scripts\activate  # En Windows
```

### 2. Instalar dependencias de Python

```bash
pip install -r requirements.txt
```

### 3. Ejecutar la aplicación localmente (modo desarrollo)

```bash
streamlit run app.py
```
