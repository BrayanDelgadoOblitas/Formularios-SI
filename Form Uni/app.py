import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

# Configuración de página
st.set_page_config(page_title="Predicción Univariable de PM2.5 con stacking", page_icon="🌫", layout="wide")

# Estilos CSS para mejorar la apariencia
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    .stFileUploader>div>div {
        background-color: white;
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 10px;
    }
    .interpretation-box {
        background-color: #482978;
        border-left: 4px solid #2196F3;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .success-box {
        background-color: #26818e;
        border-left: 4px solid #28a745;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Función para generar interpretaciones de calidad del aire
def interpretar_calidad_aire(valor):
    if valor <= 12:
        return {
            "categoria": "Buena",
            "color": "green",
            "descripcion": "La calidad del aire es satisfactoria y la contaminación del aire presenta poco o ningún riesgo.",
            "recomendaciones": "Es un buen momento para actividades al aire libre."
        }
    elif valor <= 35.4:
        return {
            "categoria": "Moderada",
            "color": "yellow", 
            "descripcion": "La calidad del aire es aceptable. Sin embargo, puede haber una preocupación moderada para un número muy pequeño de personas.",
            "recomendaciones": "Las personas inusualmente sensibles deben considerar reducir las actividades al aire libre prolongadas o pesadas."
        }
    elif valor <= 55.4:
        return {
            "categoria": "Insalubre para grupos sensibles",
            "color": "orange",
            "descripcion": "Los miembros de grupos sensibles pueden experimentar efectos en la salud.",
            "recomendaciones": "Los niños, adultos mayores y personas con problemas cardíacos o pulmonares deben evitar esfuerzos prolongados al aire libre."
        }
    elif valor <= 150.4:
        return {
            "categoria": "Insalubre",
            "color": "red",
            "descripcion": "Todos pueden comenzar a experimentar efectos en la salud.",
            "recomendaciones": "Todos deben evitar actividades prolongadas o pesadas al aire libre."
        }
    elif valor <= 250.4:
        return {
            "categoria": "Muy insalubre",
            "color": "purple",
            "descripcion": "Advertencias de salud de condiciones de emergencia. Es probable que toda la población se vea afectada.",
            "recomendaciones": "Todos deben evitar actividades al aire libre; todos deben permanecer en interiores con ventanas y puertas cerradas."
        }
    else:
        return {
            "categoria": "Peligrosa",
            "color": "maroon",
            "descripcion": "Alerta sanitaria: todos pueden experimentar efectos más graves para la salud.",
            "recomendaciones": "Todos deben evitar actividades al aire libre; todos deben permanecer en interiores con ventanas y puertas cerradas."
        }

# Función para interpretar patrones temporales
def interpretar_patron_temporal(valores):
    """Analiza patrones en los datos de PM2.5"""
    variabilidad = np.std(valores) / np.mean(valores) * 100
    
    # Análisis de picos
    media = np.mean(valores)
    desv = np.std(valores)
    picos_altos = np.sum(valores > media + 1.5 * desv)
    picos_bajos = np.sum(valores < media - 1.5 * desv)
    
    # Análisis de tendencia horaria
    horas_criticas = []
    for i, valor in enumerate(valores):
        if valor > 35.4:  # Umbral moderado
            horas_criticas.append(i + 1)
    
    interpretacion = {
        "variabilidad": variabilidad,
        "picos_altos": picos_altos,
        "picos_bajos": picos_bajos,
        "horas_criticas": horas_criticas,
        "estabilidad": "Alta" if variabilidad < 20 else "Media" if variabilidad < 40 else "Baja"
    }
    
    return interpretacion

# Título centrado
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("🌫 Predicción Univariable de PM2.5 con stacking")
    st.markdown("Sube un archivo **Excel (.xlsx)** que contenga una columna llamada **'PM2.5'** con al menos 24 valores horarios consecutivos.")

# Cargar modelos
try:
    rnn_model = load_model("RNN_modelo.h5")
    stacking_model = joblib.load("stacking_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.error(f"❌ Error al cargar los modelos: {e}")
    st.stop()

# Subida del archivo centrada
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader("📁 Subir archivo Excel", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        # Limpieza de valores no numéricos en PM2.5
        df["PM2.5"] = pd.to_numeric(df["PM2.5"], errors="coerce")

        # Validaciones
        if "PM2.5" not in df.columns:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.error("❌ El archivo debe contener una columna llamada 'PM2.5'.")
        elif df["PM2.5"].dropna().shape[0] < 24:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.error("❌ La columna 'PM2.5' debe tener al menos 24 valores numéricos válidos.")
        else:
            # Tomamos los últimos 24 datos válidos
            pm25_data = df["PM2.5"].dropna()
            pm25_values = pm25_data.values[-24:].reshape(-1, 1)
            scaled_input = scaler.transform(pm25_values).reshape(1, 24, 1)

            # Predicción con RNN
            rnn_pred = rnn_model.predict(scaled_input, verbose=0)[0][0]

            # Predicción final con stacking
            stacking_input = np.array([[rnn_pred, rnn_pred, rnn_pred]])
            stacking_output = stacking_model.predict(stacking_input)[0]

            # Desnormalizar predicción
            final_pm25 = scaler.inverse_transform([[stacking_output]])[0][0]

            # MOSTRAR RESULTADOS Y GRÁFICOS
            st.success(f"🌤 Predicción del valor PM2.5 para 24 horas después: **{final_pm25:.2f} µg/m³**")
            
            # Clasificación de calidad del aire
            calidad_info = interpretar_calidad_aire(final_pm25)
            st.markdown(f"**Calidad del aire:** <span style='color:{calidad_info['color']}'>{calidad_info['categoria']}</span>", unsafe_allow_html=True)

            # Crear pestañas para organizar los gráficos
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Serie de Tiempo", "📈 Análisis de Tendencia", "📋 Estadísticas", "🎯 Predicción vs Histórico"])

            # Datos para análisis
            historical_pm25 = pm25_values.flatten()
            patron_info = interpretar_patron_temporal(historical_pm25)
            
            with tab1:
                st.subheader("📊 Serie de Tiempo PM2.5 (Últimas 24 horas)")
                
                # Gráfico principal de serie de tiempo
                fig_ts = go.Figure()
                
                fig_ts.add_trace(go.Scatter(
                    x=list(range(1, 25)),
                    y=historical_pm25,
                    mode='lines+markers',
                    name='PM2.5 Histórico',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=8, color='#1f77b4')
                ))
                
                # Agregar líneas de referencia para calidad del aire
                fig_ts.add_hline(y=12, line_dash="dot", line_color="green", 
                                annotation_text="Buena (≤12)", annotation_position="top right")
                fig_ts.add_hline(y=35.4, line_dash="dot", line_color="yellow", 
                                annotation_text="Moderada (≤35.4)", annotation_position="top right")
                fig_ts.add_hline(y=55.4, line_dash="dot", line_color="orange", 
                                annotation_text="Insalubre GS (≤55.4)", annotation_position="top right")
                fig_ts.add_hline(y=150.4, line_dash="dot", line_color="red", 
                                annotation_text="Insalubre (≤150.4)", annotation_position="top right")
                
                fig_ts.update_layout(
                    title="PM2.5 - Últimas 24 Horas",
                    xaxis_title="Hora",
                    yaxis_title="PM2.5 (µg/m³)",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_ts, use_container_width=True)
                
                # INTERPRETACIÓN DEL GRÁFICO DE SERIE DE TIEMPO

                max_val = np.max(historical_pm25)
                min_val = np.min(historical_pm25)
                avg_val = np.mean(historical_pm25)

                # Mejorado:
                with st.container():
                    st.markdown("""
                    <div class="interpretation-box">
                    <h4>🔍 Interpretación de la Serie de Tiempo:</h4>
                    """, unsafe_allow_html=True)
                    
                    # Análisis de rango
                    st.write(f"""
                    **📊 Rango de Valores:**
                    - **Máximo:** {max_val:.1f} µg/m³ (hora {np.argmax(historical_pm25)+1})
                    - **Mínimo:** {min_val:.1f} µg/m³ (hora {np.argmin(historical_pm25)+1})
                    - **Amplitud:** {max_val-min_val:.1f} µg/m³
                    """)
                    
                    # Análisis de variabilidad
                    st.write(f"""
                    **📈 Variabilidad:**
                    - **Promedio:** {avg_val:.1f} µg/m³
                    - **Desviación estándar:** {np.std(historical_pm25):.1f} µg/m³
                    - **Coeficiente de variación:** {patron_info['variabilidad']:.1f}% ({patron_info['estabilidad']})
                    """)
                    
                    # Análisis de valores críticos
                    st.write("""
                    **⚠️ Niveles Críticos:""")
                    if patron_info['horas_criticas']:
                        horas_str = ", ".join(map(str, patron_info['horas_criticas']))
                        st.write(f"- Horas con PM2.5 >35.4 µg/m³: {horas_str}")
                        st.write("- Posibles causas: Horas pico de tráfico, condiciones de inversión térmica")
                    else:
                        st.write("- No se detectaron horas con niveles problemáticos")
                    
                    # Recomendaciones basadas en patrones
                    st.write("""
                    **🔍 Recomendaciones:**
                    - Monitorear especialmente durante las horas críticas identificadas
                    - Comparar con datos meteorológicos para identificar causas externas
                    - Validar con datos históricos si este patrón es recurrente
                    """)

                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Mostrar valores críticos
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🔺 Valor Máximo", f"{max_val:.2f} µg/m³")
                with col2:
                    st.metric("🔻 Valor Mínimo", f"{min_val:.2f} µg/m³")
                with col3:
                    st.metric("📊 Promedio", f"{avg_val:.2f} µg/m³")

            with tab2:
                st.subheader("📈 Análisis de Tendencia y Patrones")
                
                # Crear subgráficos
                col1, col2 = st.columns(2)
                
                with col1:
                    # Gráfico de barras por horas
                    fig_bars = go.Figure()
                    fig_bars.add_trace(go.Bar(
                        x=list(range(1, 25)),
                        y=historical_pm25,
                        marker_color=historical_pm25,
                        marker_colorscale='Viridis',
                        name='PM2.5'
                    ))
                    
                    fig_bars.update_layout(
                        title="PM2.5 por Hora (Barras)",
                        xaxis_title="Hora",
                        yaxis_title="PM2.5 (µg/m³)",
                        height=400
                    )
                    st.plotly_chart(fig_bars, use_container_width=True)
                    
                    # INTERPRETACIÓN DEL GRÁFICO DE BARRAS
                    st.markdown("""
                    <div class="interpretation-box">
                    <h5>📊 Análisis del Gráfico de Barras:</h5>
                    """, unsafe_allow_html=True)
                    
                    # Encontrar las horas con valores más altos y más bajos
                    hora_max = np.argmax(historical_pm25) + 1
                    hora_min = np.argmin(historical_pm25) + 1
                    
                    interpretacion_barras = f"""
                    - **Hora pico:** Hora {hora_max} con {max_val:.1f} µg/m³
                    - **Hora mínima:** Hora {hora_min} con {min_val:.1f} µg/m³
                    - **Variación:** Las barras muestran la distribución horaria de contaminación
                    - **Colores:** Verde = menor contaminación, Amarillo/Rojo = mayor contaminación
                    """
                    
                    st.markdown(interpretacion_barras)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    # Análisis de variabilidad
                    diff_values = np.diff(historical_pm25)
                    
                    fig_diff = go.Figure()
                    fig_diff.add_trace(go.Scatter(
                        x=list(range(2, 25)),
                        y=diff_values,
                        mode='lines+markers',
                        name='Cambio Hora a Hora',
                        line=dict(color='red', width=2),
                        marker=dict(size=6)
                    ))
                    
                    fig_diff.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
                    
                    fig_diff.update_layout(
                        title="Variabilidad Hora a Hora",
                        xaxis_title="Hora",
                        yaxis_title="Cambio PM2.5 (µg/m³)",
                        height=400
                    )
                    st.plotly_chart(fig_diff, use_container_width=True)
                    
                    # INTERPRETACIÓN DE VARIABILIDAD
                    st.markdown("""
                    <div class="interpretation-box">
                    <h5>📈 Análisis de Variabilidad:</h5>
                    """, unsafe_allow_html=True)
                    
                    max_cambio = np.max(np.abs(diff_values))
                    cambios_significativos = np.sum(np.abs(diff_values) > 5)
                    
                    interpretacion_variabilidad = f"""
                    - **Línea cero:** Representa estabilidad (sin cambio)
                    - **Arriba de cero:** Aumento en contaminación
                    - **Debajo de cero:** Disminución en contaminación
                    - **Mayor cambio:** {max_cambio:.1f} µg/m³ entre horas consecutivas
                    - **Cambios significativos:** {cambios_significativos} cambios >5 µg/m³
                    """
                    
                    if max_cambio > 10:
                        interpretacion_variabilidad += "\n- ⚠️ **Alerta:** Cambios bruscos detectados"
                    elif max_cambio < 3:
                        interpretacion_variabilidad += "\n- ✅ **Estable:** Cambios graduales y controlados"
                    
                    st.markdown(interpretacion_variabilidad)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Análisis de tendencia
                st.subheader("🔍 Análisis de Tendencia")
                
                # Calcular tendencia simple
                x_trend = np.arange(len(historical_pm25))
                z_trend = np.polyfit(x_trend, historical_pm25, 1)
                p_trend = np.poly1d(z_trend)
                
                fig_trend = go.Figure()
                
                # Datos originales
                fig_trend.add_trace(go.Scatter(
                    x=list(range(1, 25)),
                    y=historical_pm25,
                    mode='markers',
                    name='Datos Reales',
                    marker=dict(size=8, color='blue')
                ))
                
                # Línea de tendencia
                fig_trend.add_trace(go.Scatter(
                    x=list(range(1, 25)),
                    y=p_trend(x_trend),
                    mode='lines',
                    name='Tendencia',
                    line=dict(color='red', width=3, dash='dash')
                ))
                
                fig_trend.update_layout(
                    title="PM2.5 con Línea de Tendencia",
                    xaxis_title="Hora",
                    yaxis_title="PM2.5 (µg/m³)",
                    height=400
                )
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # Mostrar interpretación de tendencia
                pendiente = z_trend[0]
                
                # INTERPRETACIÓN DETALLADA DE TENDENCIA
                st.markdown("""
                <div class="interpretation-box">
                <h4>📊 Interpretación de la Tendencia:</h4>
                """, unsafe_allow_html=True)
                
                if pendiente > 0.5:
                    tendencia_texto = "📈 **Tendencia ASCENDENTE** - Los valores están aumentando"
                    tendencia_color = "red"
                    interpretacion_detallada = f"""
                    - **Dirección:** Ascendente (empeoramiento)
                    - **Velocidad:** {pendiente:.3f} µg/m³ por hora
                    - **Implicación:** La contaminación tiende a aumentar con el tiempo
                    - **Recomendación:** Monitorear de cerca y tomar precauciones
                    """
                elif pendiente < -0.5:
                    tendencia_texto = "📉 **Tendencia DESCENDENTE** - Los valores están disminuyendo"
                    tendencia_color = "green"
                    interpretacion_detallada = f"""
                    - **Dirección:** Descendente (mejoramiento)
                    - **Velocidad:** {abs(pendiente):.3f} µg/m³ por hora de mejora
                    - **Implicación:** La calidad del aire tiende a mejorar
                    - **Perspectiva:** Condiciones favorables en desarrollo
                    """
                else:
                    tendencia_texto = "➡️ **Tendencia ESTABLE** - Los valores se mantienen relativamente constantes"
                    tendencia_color = "blue"
                    interpretacion_detallada = f"""
                    - **Dirección:** Estable (sin cambio significativo)
                    - **Variación:** ±{abs(pendiente):.3f} µg/m³ por hora
                    - **Implicación:** Condiciones atmosféricas estables
                    - **Perspectiva:** Comportamiento predecible
                    """
                
                st.markdown(f"<span style='color:{tendencia_color}'>{tendencia_texto}</span>", unsafe_allow_html=True)
                st.markdown(interpretacion_detallada)
                st.markdown("</div>", unsafe_allow_html=True)

            with tab3:
                st.subheader("📋 Estadísticas Descriptivas PM2.5")
                
                # Calcular estadísticas
                stats_data = {
                    'Estadística': ['Promedio', 'Mediana', 'Desviación Estándar', 'Varianza', 'Mínimo', 'Máximo', 'Rango'],
                    'Valor': [
                        np.mean(historical_pm25),
                        np.median(historical_pm25),
                        np.std(historical_pm25),
                        np.var(historical_pm25),
                        np.min(historical_pm25),
                        np.max(historical_pm25),
                        np.max(historical_pm25) - np.min(historical_pm25)
                    ]
                }
                
                # Mostrar métricas principales
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Promedio", f"{np.mean(historical_pm25):.2f} µg/m³")
                    st.write(f"**Mediana:** {np.median(historical_pm25):.2f}")
                
                with col2:
                    st.metric("Desv. Estándar", f"{np.std(historical_pm25):.2f}")
                    st.write(f"**Varianza:** {np.var(historical_pm25):.2f}")
                
                with col3:
                    st.metric("Mínimo", f"{np.min(historical_pm25):.2f} µg/m³")
                    st.write(f"**Máximo:** {np.max(historical_pm25):.2f}")
                
                with col4:
                    st.metric("Rango", f"{np.max(historical_pm25) - np.min(historical_pm25):.2f}")
                    cv = (np.std(historical_pm25) / np.mean(historical_pm25)) * 100
                    st.write(f"**Coef. Variación:** {cv:.1f}%")
                
                # INTERPRETACIÓN DE ESTADÍSTICAS
                st.markdown("""
                <div class="interpretation-box">
                <h4>📊 Interpretación de las Estadísticas:</h4>
                """, unsafe_allow_html=True)
                
                promedio = np.mean(historical_pm25)
                mediana = np.median(historical_pm25)
                desv_std = np.std(historical_pm25)
                cv = (desv_std / promedio) * 100
                
                interpretacion_stats = f"""
                **Medidas de Tendencia Central:**
                - **Promedio vs Mediana:** """
                
                if abs(promedio - mediana) < 2:
                    interpretacion_stats += "Valores similares indican distribución simétrica"
                elif promedio > mediana:
                    interpretacion_stats += "Promedio > Mediana indica algunos valores altos extremos"
                else:
                    interpretacion_stats += "Mediana > Promedio indica algunos valores bajos extremos"
                
                interpretacion_stats += f"""
                
                **Medidas de Dispersión:**
                - **Coeficiente de Variación ({cv:.1f}%):** """
                
                if cv < 15:
                    interpretacion_stats += "Baja variabilidad - Datos consistentes"
                elif cv < 30:
                    interpretacion_stats += "Variabilidad moderada - Fluctuaciones normales"
                else:
                    interpretacion_stats += "Alta variabilidad - Datos muy dispersos"
                
                interpretacion_stats += f"""
                - **Rango ({np.max(historical_pm25) - np.min(historical_pm25):.1f} µg/m³):** """
                
                rango = np.max(historical_pm25) - np.min(historical_pm25)
                if rango < 10:
                    interpretacion_stats += "Rango pequeño - Condiciones estables"
                elif rango < 25:
                    interpretacion_stats += "Rango moderado - Variación normal"
                else:
                    interpretacion_stats += "Rango amplio - Condiciones muy variables"
                
                st.markdown(interpretacion_stats)
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Tabla de estadísticas completa
                stats_df = pd.DataFrame(stats_data)
                stats_df['Valor'] = stats_df['Valor'].round(3)
                st.subheader("📊 Tabla de Estadísticas Completa")
                st.dataframe(stats_df, use_container_width=True)
                
                # Histograma
                st.subheader("📊 Distribución de Valores PM2.5")
                
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=historical_pm25,
                    nbinsx=8,
                    marker_color='skyblue',
                    marker_line_color='black',
                    marker_line_width=1,
                    opacity=0.7
                ))
                
                fig_hist.update_layout(
                    title="Distribución de Valores PM2.5",
                    xaxis_title="PM2.5 (µg/m³)",
                    yaxis_title="Frecuencia",
                    height=400
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # INTERPRETACIÓN DEL HISTOGRAMA
                st.markdown("""
                <div class="interpretation-box">
                <h5>📈 Interpretación del Histograma:</h5>
                """, unsafe_allow_html=True)
                
                # Análisis de la distribución
                q1 = np.percentile(historical_pm25, 25)
                q3 = np.percentile(historical_pm25, 75)
                
                interpretacion_hist = f"""
                - **Forma de la distribución:** El histograma muestra cómo se distribuyen los valores
                - **Concentración de datos:** El 50% central de los datos está entre {q1:.1f} y {q3:.1f} µg/m³
                - **Barras más altas:** Indican los rangos de valores más frecuentes
                - **Barras aisladas:** Pueden indicar valores atípicos o condiciones especiales
                """
                
                # Análisis de normalidad básico
                if abs(promedio - mediana) < desv_std * 0.5:
                    interpretacion_hist += "\n- **Distribución:** Aproximadamente normal (simétrica)"
                else:
                    interpretacion_hist += "\n- **Distribución:** Asimétrica (sesgada)"
                
                st.markdown(interpretacion_hist)
                st.markdown("</div>", unsafe_allow_html=True)

            with tab4:
                st.subheader("🎯 Predicción vs Valores Históricos")
                
                # Crear datos para el gráfico de predicción
                time_points = list(range(-24, 1))  # -24 a 0 para histórico, 1 para predicción
                
                fig_pred = go.Figure()
                
                # Línea histórica
                fig_pred.add_trace(go.Scatter(
                    x=time_points[:-1],
                    y=historical_pm25,
                    mode='lines+markers',
                    name='Valores Históricos',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8)
                ))
                
                # Punto de predicción
                fig_pred.add_trace(go.Scatter(
                    x=[1],
                    y=[final_pm25],
                    mode='markers',
                    name='Predicción (+24h)',
                    marker=dict(color='red', size=20, symbol='star')
                ))
                
                # Línea de conexión
                fig_pred.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[historical_pm25[-1], final_pm25],
                    mode='lines',
                    name='Proyección',
                    line=dict(color='red', width=3, dash='dash')
                ))
                
                # Agregar líneas de referencia para calidad del aire
                fig_pred.add_hline(y=12, line_dash="dot", line_color="green", 
                                  annotation_text="Buena (≤12)")
                fig_pred.add_hline(y=35.4, line_dash="dot", line_color="yellow", 
                                  annotation_text="Moderada (≤35.4)")
                fig_pred.add_hline(y=55.4, line_dash="dot", line_color="orange", 
                                  annotation_text="Insalubre GS (≤55.4)")
                
                fig_pred.update_layout(
                    title="PM2.5: Histórico vs Predicción",
                    xaxis_title="Tiempo (horas desde ahora)",
                    yaxis_title="PM2.5 (µg/m³)",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # INTERPRETACIÓN DETALLADA DE LA PREDICCIÓN
                st.markdown("""
                <div class="interpretation-box">
                <h4>🎯 Interpretación de la Predicción:</h4>
                """, unsafe_allow_html=True)
                
                cambio_absoluto = final_pm25 - historical_pm25[-1]
                cambio_porcentual = (cambio_absoluto / historical_pm25[-1]) * 100
                
                interpretacion_pred = f"""
                **Análisis del Gráfico de Predicción:**
                - **Línea azul:** Representa los 24 valores históricos reales
                - **Estrella roja:** Muestra la predicción para las próximas 24 horas
                - **Línea discontinua roja:** Conecta el último valor real con la predicción
                - **Líneas horizontales:** Muestran los umbrales de calidad del aire
                
                **Contexto de la Predicción:**
                - **Valor actual:** {historical_pm25[-1]:.2f} µg/m³
                - **Valor predicho:** {final_pm25:.2f} µg/m³
                - **Cambio absoluto:** {cambio_absoluto:+.2f} µg/m³
                - **Cambio porcentual:** {cambio_porcentual:+.1f}%
                """
                
                # Análisis de cruce de umbrales
                calidad_actual = interpretar_calidad_aire(historical_pm25[-1])
                calidad_predicha = interpretar_calidad_aire(final_pm25)
                
                if calidad_actual['categoria'] != calidad_predicha['categoria']:
                    interpretacion_pred += f"""
                    
                    ⚠️ **CAMBIO DE CATEGORÍA DETECTADO:**
                    - **Actual:** {calidad_actual['categoria']} → **Predicho:** {calidad_predicha['categoria']}
                    """
                
                st.markdown(interpretacion_pred)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Comparación numérica
                st.subheader("📊 Comparación Numérica")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="PM2.5 Actual (última medición)",
                        value=f"{historical_pm25[-1]:.2f} µg/m³"
                    )
                
                with col2:
                    st.metric(
                        label="PM2.5 Predicho (+24h)",
                        value=f"{final_pm25:.2f} µg/m³",
                        delta=f"{cambio_absoluto:.2f}"
                    )
                
                with col3:
                    st.metric(
                        label="Cambio Porcentual",
                        value=f"{cambio_porcentual:.1f}%"
                    )
                
                with col4:
                    promedio_historico = np.mean(historical_pm25)
                    vs_promedio = ((final_pm25 - promedio_historico) / promedio_historico) * 100
                    st.metric(
                        label="vs Promedio Histórico",
                        value=f"{vs_promedio:+.1f}%"
                    )
                
                # Interpretación del cambio
                st.subheader("🔍 Interpretación del Cambio Predicho")
                
                # Determinar el tipo de caja de interpretación según el cambio
                if abs(cambio_porcentual) < 5:
                    interpretacion = "✅ **Cambio MÍNIMO** - Se espera que los niveles se mantengan similares"
                    caja_clase = "success-box"
                elif cambio_porcentual > 5:
                    if cambio_porcentual > 20:
                        interpretacion = "🚨 **Aumento SIGNIFICATIVO** - Se espera un deterioro notable de la calidad del aire"
                        caja_clase = "danger-box"
                    else:
                        interpretacion = "⚠️ **Aumento MODERADO** - Se espera un ligero deterioro de la calidad del aire"
                        caja_clase = "warning-box"
                else:  # cambio_porcentual < -5
                    if cambio_porcentual < -20:
                        interpretacion = "🌟 **Mejora SIGNIFICATIVA** - Se espera una notable mejora de la calidad del aire"
                        caja_clase = "success-box"
                    else:
                        interpretacion = "📈 **Mejora MODERADA** - Se espera una ligera mejora de la calidad del aire"
                        caja_clase = "success-box"
                
                st.markdown(f"""
                <div class="{caja_clase}">
                <h4>{interpretacion}</h4>
                """, unsafe_allow_html=True)
                
                # Interpretación detallada adicional
                interpretacion_detallada = f"""
                **Factores a Considerar:**
                - **Confiabilidad:** La predicción se basa en patrones históricos de las últimas 24 horas
                - **Contexto temporal:** Las condiciones meteorológicas pueden influir significativamente
                - **Margen de error:** Las predicciones de calidad del aire tienen inherente incertidumbre
                
                **Recomendaciones basadas en la predicción:**
                """
                
                # Recomendaciones específicas según la calidad predicha
                if final_pm25 <= 12:
                    interpretacion_detallada += """
                    - ✅ Excelente momento para actividades al aire libre
                    - ✅ Ventilación natural recomendada
                    - ✅ Ejercicio al aire libre sin restricciones
                    """
                elif final_pm25 <= 35.4:
                    interpretacion_detallada += """
                    - ✅ Actividades al aire libre generalmente seguras
                    - ⚠️ Personas sensibles deben estar atentas
                    - ✅ Ventilación con precaución
                    """
                elif final_pm25 <= 55.4:
                    interpretacion_detallada += """
                    - ⚠️ Grupos sensibles deben limitar actividades prolongadas al aire libre
                    - ⚠️ Considerar usar mascarillas en exteriores
                    - ⚠️ Limitar ventilación natural
                    """
                elif final_pm25 <= 150.4:
                    interpretacion_detallada += """
                    - 🚨 Evitar actividades prolongadas al aire libre
                    - 🚨 Usar purificadores de aire en interiores
                    - 🚨 Mascarillas recomendadas para salidas esenciales
                    """
                else:
                    interpretacion_detallada += """
                    - 🚨 Evitar completamente actividades al aire libre
                    - 🚨 Mantener puertas y ventanas cerradas
                    - 🚨 Usar purificadores de aire y mascarillas N95
                    """
                
                # Análisis de precisión basado en la estabilidad histórica
                if patron_info['variabilidad'] < 20:
                    interpretacion_detallada += f"""
                    
                    **Confianza en la predicción:** ALTA
                    - Los datos históricos muestran baja variabilidad ({patron_info['variabilidad']:.1f}%)
                    - Patrones estables indican predicción más confiable
                    """
                elif patron_info['variabilidad'] < 40:
                    interpretacion_detallada += f"""
                    
                    **Confianza en la predicción:** MODERADA
                    - Los datos históricos muestran variabilidad moderada ({patron_info['variabilidad']:.1f}%)
                    - Considerar factores externos adicionales
                    """
                else:
                    interpretacion_detallada += f"""
                    
                    **Confianza en la predicción:** BAJA
                    - Los datos históricos muestran alta variabilidad ({patron_info['variabilidad']:.1f}%)
                    - Predicción menos confiable debido a patrones irregulares
                    """
                
                st.markdown(interpretacion_detallada)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Resumen final con información de salud
                st.subheader("🏥 Resumen de Impacto en Salud")
                
                calidad_pred_info = interpretar_calidad_aire(final_pm25)
                
                if calidad_pred_info['color'] in ['green', 'yellow']:
                    resumen_clase = "success-box"
                elif calidad_pred_info['color'] == 'orange':
                    resumen_clase = "warning-box"
                else:
                    resumen_clase = "danger-box"
                
                st.markdown(f"""
                <div class="{resumen_clase}">
                <h4>📋 Resumen para las Próximas 24 Horas:</h4>
                <p><strong>Calidad del Aire Predicha:</strong> {calidad_pred_info['categoria']}</p>
                <p><strong>Descripción:</strong> {calidad_pred_info['descripcion']}</p>
                <p><strong>Recomendaciones:</strong> {calidad_pred_info['recomendaciones']}</p>
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.error(f"❌ Error al procesar el archivo: {e}")