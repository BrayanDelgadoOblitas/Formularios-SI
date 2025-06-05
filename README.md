# Proyecto de Predicción de Calidad del Aire

## 📌 Descripción

Este proyecto consta de dos formularios (`Form Uni` y `Form Multi`) diseñados para cargar datos ambientales desde un archivo **Excel** y predecir el **Índice de Calidad del Aire (AQI)** para las **próximas 24 horas**.

- **Form Uni**: Predicción basada únicamente en **PM2.5**.
- **Form Multi**: Predicción basada en **PM2.5, PM10, Temperatura, Humedad y NO2**, ofreciendo un análisis más completo.

Ambos formularios realizan:
- ✅ **Predicciones de calidad del aire**
- ✅ **Gráficos interactivos** de tendencias
- ✅ **Interpretación automática** de resultados

## 📥 Requisitos

- **Microsoft Excel** (2016 o superior recomendado)
- **Archivo de entrada** con al menos **24 registros horarios**
- **Estructura de columnas específica** (ver más abajo)

## 📂 Estructura del Archivo Excel

### 🔹 Para `Form Uni` (Predicción con PM2.5)

| Fecha/Hora | PM2.5 |
|------------|---------------|
| 2024-01-01 00:00 | 12.5 |
| 2024-01-01 01:00 | 14.2 |
| ... (mínimo 24 filas) | ... |

### 🔹 Para `Form Multi` (Predicción multivariable)

| Fecha/Hora | PM2.5 | PM10 | Temperatura | Humedad | NO2 |
|------------|-------|------|-------------|---------|-----|
| 2024-01-01 00:00 | 12.5 | 25.1 | 22.0 | 65 | 18 |
| 2024-01-01 01:00 | 14.2 | 28.3 | 21.5 | 68 | 20 |
| ... (mínimo 24 filas) | ... | ... | ... | ... | ... |

⚠️ **Importante**:
- El archivo **debe contener al menos 24 registros** (horas) para garantizar una predicción precisa
- Las **fechas deben ser consecutivas** y en formato `YYYY-MM-DD HH:MM`

## 🚀 Instrucciones de Uso

1. **Navegue** a la subcarpeta del formulario que desea utilizar (`Form Uni` o `Form Multi`)
2. **Consulte el README** específico de cada subcarpeta para obtener instrucciones detalladas de ejecución
3. **Prepare su archivo de datos** siguiendo la estructura requerida mencionada arriba
4. **Ejecute el formulario** siguiendo las instrucciones específicas de cada subcarpeta
5. **Revise los resultados**:
   - 📊 **Gráficos** de tendencia histórica y predicción
   - 📝 **Interpretación automática** de los niveles de contaminación

## 📁 Estructura del Repositorio

Cada subcarpeta del repositorio contiene:
- Un formulario específico para predicción de calidad del aire
- Un README con instrucciones detalladas para ejecutar ese formulario particular
- Archivos necesarios para el funcionamiento del formulario

## ⚠️ Limitaciones

- **Solo predice 24 horas** hacia adelante
- La precisión depende de la **calidad y cantidad de datos** (mínimo 24 registros)
- No incluye variables como **SO2, CO u O3** (solo disponibles las variables mencionadas en `Form Multi`)
- El modelo está optimizado para **ciudades con patrones de contaminación similares**