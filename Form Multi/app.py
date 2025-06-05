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
st.set_page_config(page_title="Predicción Multivariable de PM2.5", page_icon="🌫", layout="wide")

# Estilo visual mejorado
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

# Función para analizar patrones multivariados
def analizar_patrones_multivariados(data_24h):
    """Analiza patrones en los datos multivariados"""
    analisis = {}
    
    for var in data_24h.columns:
        valores = data_24h[var].values
        variabilidad = np.std(valores) / np.mean(valores) * 100
        
        # Análisis de picos
        media = np.mean(valores)
        desv = np.std(valores)
        picos_altos = np.sum(valores > media + 1.5 * desv)
        picos_bajos = np.sum(valores < media - 1.5 * desv)
        
        analisis[var] = {
            "variabilidad": variabilidad,
            "picos_altos": picos_altos,
            "picos_bajos": picos_bajos,
            "estabilidad": "Alta" if variabilidad < 20 else "Media" if variabilidad < 40 else "Baja",
            "promedio": media,
            "maximo": np.max(valores),
            "minimo": np.min(valores)
        }
    
    return analisis

# Función para interpretar correlaciones
def interpretar_correlaciones(corr_matrix):
    """Interpreta la matriz de correlación"""
    interpretaciones = []
    
    # Buscar correlaciones significativas con PM2.5
    pm25_correlations = corr_matrix['PM2.5'].drop('PM2.5')
    
    for var, corr in pm25_correlations.items():
        if abs(corr) > 0.7:
            fuerza = "muy fuerte"
        elif abs(corr) > 0.5:
            fuerza = "fuerte"
        elif abs(corr) > 0.3:
            fuerza = "moderada"
        elif abs(corr) > 0.1:
            fuerza = "débil"
        else:
            fuerza = "muy débil"
        
        direccion = "positiva" if corr > 0 else "negativa"
        
        interpretaciones.append({
            "variable": var,
            "correlacion": corr,
            "fuerza": fuerza,
            "direccion": direccion
        })
    
    return interpretaciones

# Título centrado
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("🌫 Predicción Multivariable de PM2.5 con Stacking")
    st.markdown("Sube un archivo **Excel (.xlsx)** con al menos 24 registros y las siguientes columnas:")
    st.markdown("**PM2.5, PM10, Temperatura, Humedad, NO2**")

# Cargar modelos
try:
    rnn_model = load_model("RNN_modelo_multivariable.h5")
    stacking_model = joblib.load("stacking_model.pkl")
    scalers = joblib.load("scalers.pkl")
except Exception as e:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.error(f"❌ Error al cargar los modelos: {e}")
    st.stop()

# Columnas necesarias
variables = ["PM2.5", "PM10", "Temperatura", "Humedad", "NO2"]

# Cargar archivo
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader("📁 Subir archivo Excel", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        # Verificar existencia de columnas
        missing = [v for v in variables if v not in df.columns]
        if missing:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.error(f"❌ Faltan las siguientes columnas: {', '.join(missing)}")
        else:
            # Convertir todas las columnas a numéricas (con coerción)
            for col in variables:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Filas completas válidas (sin NaNs)
            df_valid = df[variables].dropna()
            if len(df_valid) < 24:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.error("❌ Se requieren al menos 24 filas válidas con datos completos.")
            else:
                # Tomar las últimas 24 filas válidas
                data_24h = df_valid.iloc[-24:]

                # Escalar por variable
                scaled = []
                for var in variables:
                    scaled_var = scalers[var].transform(data_24h[[var]])
                    scaled.append(scaled_var)

                # Convertir a (1, 24, 5)
                input_scaled = np.concatenate(scaled, axis=1).reshape(1, 24, len(variables))

                # Predicción con RNN
                rnn_pred = rnn_model.predict(input_scaled, verbose=0)[0][0]

                # Predicción stacking (dummy con rnn_pred 3 veces si no tienes otros modelos)
                stacking_input = np.array([[rnn_pred, rnn_pred, rnn_pred]])
                stacking_output = stacking_model.predict(stacking_input)[0]

                # Desescalar predicción final
                final_pm25 = scalers["PM2.5"].inverse_transform([[stacking_output]])[0][0]

                # Análisis de patrones multivariados
                analisis_patrones = analizar_patrones_multivariados(data_24h)

                # MOSTRAR RESULTADOS Y GRÁFICOS
                st.success(f"🌤 Predicción del valor PM2.5 para 24 horas después: **{final_pm25:.2f} µg/m³**")
                
                # Clasificación de calidad del aire
                calidad_info = interpretar_calidad_aire(final_pm25)
                st.markdown(f"**Calidad del aire:** <span style='color:{calidad_info['color']}'>{calidad_info['categoria']}</span>", unsafe_allow_html=True)

                # Crear pestañas para organizar los gráficos
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Series de Tiempo", "📈 Correlaciones", "📋 Estadísticas", "🎯 Predicción vs Histórico"])

                with tab1:
                    st.subheader("📊 Series de Tiempo de las Variables (Últimas 24 horas)")
                    
                    # Gráfico interactivo con Plotly
                    fig = make_subplots(
                        rows=3, cols=2,
                        subplot_titles=variables,
                        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                               [{"secondary_y": False}, {"secondary_y": False}],
                               [{"secondary_y": False}, None]]
                    )
                    
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                    positions = [(1,1), (1,2), (2,1), (2,2), (3,1)]
                    
                    for i, var in enumerate(variables):
                        row, col = positions[i]
                        fig.add_trace(
                            go.Scatter(
                                x=list(range(24)),
                                y=data_24h[var].values,
                                mode='lines+markers',
                                name=var,
                                line=dict(color=colors[i], width=2),
                                marker=dict(size=6)
                            ),
                            row=row, col=col
                        )
                    
                    fig.update_layout(height=800, showlegend=False, title_text="Variables Ambientales - Últimas 24 horas")
                    fig.update_xaxes(title_text="Horas")
                    st.plotly_chart(fig, use_container_width=True)

                    # INTERPRETACIÓN DE SERIES DE TIEMPO
                    with st.container():                       
                        st.markdown("""<div class="interpretation-box">
                                    <h4>🔍 Interpretación de la Serie de Tiempo:</h4>""", unsafe_allow_html=True)
                        
                        st.write("**Análisis por Variable:**")
                        
                        for var in variables:
                            datos_var = analisis_patrones[var]
                            st.write(f"""
                            **{var}:**
                            - Rango: {datos_var['minimo']:.1f} - {datos_var['maximo']:.1f}
                            - Promedio: {datos_var['promedio']:.1f}
                            - Estabilidad: {datos_var['estabilidad']} (variabilidad: {datos_var['variabilidad']:.1f}%)
                            """)
                            
                            if datos_var['picos_altos'] > 0:
                                st.write(f" - ⚠️ {datos_var['picos_altos']} picos anómalos detectados")
                        
                        st.write(""" 
                        **Patrones Temporales Observados:**
                        - Las gráficas muestran la evolución horaria de cada variable ambiental
                        - Los patrones pueden revelar ciclos diarios (ej: temperatura) o eventos de contaminación
                        - La sincronización entre variables puede indicar relaciones causales
                        """)
                        
                        # Análisis de PM2.5 específico
                        pm25_data = data_24h['PM2.5'].values
                        valores_criticos = np.sum(pm25_data > 35.4)
                        
                        if valores_criticos > 0:
                            st.write(f"\n- 🚨 **PM2.5 Crítico:** {valores_criticos} horas superaron niveles moderados")
                        else:
                            st.write(f"\n- ✅ **PM2.5 Aceptable:** Todos los valores dentro de rangos seguros")
                        
                        st.markdown("</div>", unsafe_allow_html=True)

                    # Métricas rápidas por variable
                    st.subheader("📈 Resumen por Variable")
                    cols = st.columns(len(variables))
                    
                    for i, var in enumerate(variables):
                        with cols[i]:
                            datos_var = analisis_patrones[var]
                            st.metric(
                                label=f"{var}",
                                value=f"{datos_var['promedio']:.1f}",
                                delta=f"±{datos_var['variabilidad']:.0f}%"
                            )
                            
                            # Indicador de estabilidad con color
                            if datos_var['estabilidad'] == 'Alta':
                                st.markdown("🟢 **Estable**")
                            elif datos_var['estabilidad'] == 'Media':
                                st.markdown("🟡 **Moderada**")
                            else:
                                st.markdown("🔴 **Variable**")

                with tab2:
                    st.subheader("📈 Matriz de Correlación")
                    
                    # Matriz de correlación
                    corr_matrix = data_24h.corr()
                    interpretaciones_corr = interpretar_correlaciones(corr_matrix)
                    
                    # Heatmap con Plotly
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu',
                        zmid=0,
                        text=np.round(corr_matrix.values, 2),
                        texttemplate="%{text}",
                        textfont={"size": 12},
                        hoverongaps=False
                    ))
                    
                    fig_corr.update_layout(
                        title="Matriz de Correlación entre Variables",
                        height=500
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)

                    # INTERPRETACIÓN DE CORRELACIONES
                    with st.container():
                        st.markdown("""<div class="interpretation-box">
                                    <h4>🔍 Interpretación de la Matriz de Correlación:</h4>""", unsafe_allow_html=True)
                        
                        st.write("""
                        **Cómo leer la matriz:**
                        - **Valores cercanos a +1:** Correlación positiva fuerte (cuando una sube, la otra también)
                        - **Valores cercanos a -1:** Correlación negativa fuerte (cuando una sube, la otra baja)
                        - **Valores cercanos a 0:** Sin correlación aparente
                        - **Colores:** Azul = correlación positiva, Rojo = correlación negativa
                        
                        **Relaciones de PM2.5 con otras variables:**
                        """)
                        
                        for interp in interpretaciones_corr:
                            var = interp['variable']
                            corr = interp['correlacion']
                            fuerza = interp['fuerza']
                            direccion = interp['direccion']
                            
                            if direccion == 'positiva':
                                simbolo = "📈"
                            else:
                                simbolo = "📉"
                            
                            st.write(f"""
                            - **{var}:** {simbolo} Correlación {direccion} {fuerza} ({corr:.3f})
                            """)
                            
                            # Interpretación específica por variable
                            if var == 'PM10' and abs(corr) > 0.5:
                                st.write(" - *Material particulado relacionado*")
                            elif var == 'Temperatura' and corr < -0.3:
                                st.write(" - *Mayor temperatura puede dispersar contaminantes*")
                            elif var == 'Humedad' and abs(corr) > 0.3:
                                st.write(" - *La humedad afecta la formación de partículas*")
                            elif var == 'NO2' and corr > 0.3:
                                st.write(" - *Fuentes de combustión comunes*")
                        
                        st.markdown("</div>", unsafe_allow_html=True)

                    # Scatter plots con interpretaciones
                    st.subheader("🔍 Relación PM2.5 vs Otras Variables")
                    col1, col2 = st.columns(2)
                    
                    scatter_vars = ["PM10", "Temperatura", "Humedad", "NO2"]
                    cols = [col1, col2, col1, col2]
                    
                    for i, var in enumerate(scatter_vars):
                        with cols[i]:
                            fig_scatter = px.scatter(
                                data_24h, x=var, y="PM2.5",
                                title=f"PM2.5 vs {var}",
                                trendline="ols"
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True)
                            
                            # Interpretación específica del scatter plot
                            corr_val = corr_matrix.loc['PM2.5', var]
                            
                            if abs(corr_val) > 0.5:
                                st.markdown(f"**Relación fuerte** (r={corr_val:.2f})")
                            elif abs(corr_val) > 0.3:
                                st.markdown(f"**Relación moderada** (r={corr_val:.2f})")
                            else:
                                st.markdown(f"**Relación débil** (r={corr_val:.2f})")

                with tab3:
                    st.subheader("📋 Estadísticas Descriptivas")
                    
                    # Métricas en columnas usando componentes nativos de Streamlit
                    cols = st.columns(len(variables))
                    stats = data_24h.describe()
                    
                    for i, var in enumerate(variables):
                        with cols[i]:
                            st.markdown(f"**{var}**")
                            st.metric(
                                label="Promedio",
                                value=f"{stats.loc['mean', var]:.2f}"
                            )
                            st.write(f"**Mínimo:** {stats.loc['min', var]:.2f}")
                            st.write(f"**Máximo:** {stats.loc['max', var]:.2f}")
                            st.write(f"**Desv. Est:** {stats.loc['std', var]:.2f}")
                            
                            # Coeficiente de variación
                            cv = (stats.loc['std', var] / stats.loc['mean', var]) * 100
                            st.write(f"**CV:** {cv:.1f}%")
                            st.markdown("---")
                    
                    # INTERPRETACIÓN DE ESTADÍSTICAS
                    with st.container():
                        st.markdown("""<div class="interpretation-box">
                                    <h4>📊 Interpretación de las Estadísticas:</h4>""", unsafe_allow_html=True)
                        
                        st.write("""
                        **Análisis de Variabilidad por Variable:**
                        """)
                        
                        for var in variables:
                            promedio = stats.loc['mean', var]
                            desv_std = stats.loc['std', var]
                            cv = (desv_std / promedio) * 100
                            
                            st.write(f"""
                            **{var}:**
                            - Coeficiente de Variación: {cv:.1f}%
                            """)
                            
                            if cv < 15:
                                st.write(" - ✅ **Muy estable** (baja variabilidad)")
                            elif cv < 30:
                                st.write(" - 🟡 **Moderadamente variable**")
                            else:
                                st.write(" - 🔴 **Altamente variable**")
                        
                        st.write("""
                        **Implicaciones para la Predicción:**
                        - Variables más estables proporcionan patrones más predecibles
                        - Alta variabilidad puede indicar influencia de factores externos
                        - La combinación de variables estables e inestables enriquece el modelo
                        """)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Tabla de estadísticas completa
                    st.subheader("📊 Tabla de Estadísticas Completa")
                    st.dataframe(stats.round(2), use_container_width=True)
                    
                    # Histogramas con interpretaciones
                    st.subheader("📊 Distribución de Variables")
                    fig_hist = make_subplots(
                        rows=2, cols=3,
                        subplot_titles=variables,
                        specs=[[{}, {}, {}], [{}, {}, None]]
                    )
                    
                    positions_hist = [(1,1), (1,2), (1,3), (2,1), (2,2)]
                    
                    for i, var in enumerate(variables):
                        row, col = positions_hist[i]
                        fig_hist.add_trace(
                            go.Histogram(
                                x=data_24h[var],
                                name=var,
                                nbinsx=10,
                                marker_color=colors[i],
                                opacity=0.7
                            ),
                            row=row, col=col
                        )
                    
                    fig_hist.update_layout(height=600, showlegend=False, title_text="Distribución de Variables")
                    st.plotly_chart(fig_hist, use_container_width=True)

                    # INTERPRETACIÓN DE HISTOGRAMAS
                    with st.container():
                        st.markdown("""<div class="interpretation-box">
                                    <h4>📈 Interpretación de las Distribuciones:</h4>""", unsafe_allow_html=True)
                        
                        st.write("""
                        **Análisis de Distribuciones:**
                        - **Forma de las barras:** Indica cómo se distribuyen los valores de cada variable
                        - **Barras altas:** Representan valores más frecuentes
                        - **Barras dispersas:** Pueden indicar valores atípicos o condiciones especiales
                        
                        **Patrones por Variable:**
                        """)
                        
                        for var in variables:
                            q1 = np.percentile(data_24h[var], 25)
                            q3 = np.percentile(data_24h[var], 75)
                            mediana = np.median(data_24h[var])
                            promedio = np.mean(data_24h[var])
                            
                            st.write(f"""
                            **{var}:**
                            - 50% central de datos: {q1:.1f} - {q3:.1f}
                            """)
                            
                            if abs(promedio - mediana) < np.std(data_24h[var]) * 0.5:
                                st.write(" - Distribución aproximadamente simétrica")
                            else:
                                st.write(" - Distribución asimétrica")
                        
                        st.markdown("</div>", unsafe_allow_html=True)

                with tab4:
                    st.subheader("🎯 Predicción vs Valores Históricos")
                    
                    # Crear datos para el gráfico de predicción
                    historical_pm25 = data_24h["PM2.5"].values
                    time_points = list(range(-24, 1))  # -24 a 0 para histórico, 1 para predicción
                    
                    # Valores históricos + predicción
                    all_values = list(historical_pm25) + [final_pm25]
                    
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
                    
                    fig_pred.update_layout(
                        title="PM2.5: Histórico vs Predicción",
                        xaxis_title="Tiempo (horas desde ahora)",
                        yaxis_title="PM2.5 (µg/m³)",
                        height=500,
                        hovermode='x unified'
                    )
                    
                    # Agregar líneas de referencia para calidad del aire
                    fig_pred.add_hline(y=12, line_dash="dot", line_color="green", 
                                      annotation_text="Buena (≤12)")
                    fig_pred.add_hline(y=35.4, line_dash="dot", line_color="yellow", 
                                      annotation_text="Moderada (≤35.4)")
                    fig_pred.add_hline(y=55.4, line_dash="dot", line_color="orange", 
                                      annotation_text="Insalubre GS (≤55.4)")
                    
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # INTERPRETACIÓN DETALLADA DE LA PREDICCIÓN MULTIVARIABLE
                    with st.container():
                        st.markdown("""<div class="interpretation-box">
                                    <h4>🎯 Interpretación de la Predicción Multivariable:</h4>""", unsafe_allow_html=True)
                        
                        cambio_absoluto = final_pm25 - historical_pm25[-1]
                        cambio_porcentual = (cambio_absoluto / historical_pm25[-1]) * 100
                        
                        # Análisis de confiabilidad basado en variables múltiples
                        variables_estables = sum(1 for var in variables if analisis_patrones[var]['estabilidad'] == 'Alta')
                        correlaciones_significativas = len([i for i in interpretaciones_corr if abs(i['correlacion']) > 0.3])
                        
                        st.write(f"""
                        **Ventaja del Modelo Multivariable:**
                        - **Variables consideradas:** PM10, Temperatura, Humedad, NO2 además de PM2.5
                        - **Variables estables:** {variables_estables}/{len(variables)} variables muestran alta estabilidad
                        - **Correlaciones significativas:** {correlaciones_significativas} variables tienen correlación moderada/fuerte con PM2.5
                        """)
                        
                        st.write(f"""
                        **Análisis del Cambio Predicho:**
                        - **Cambio absoluto:** {cambio_absoluto:+.2f} µg/m³
                        - **Cambio porcentual:** {cambio_porcentual:+.1f}%
                        """)
                        
                        # Interpretación del cambio
                        if abs(cambio_porcentual) < 5:
                            st.write("\n- 🟢 **Cambio mínimo:** La predicción sugiere condiciones estables")
                        elif abs(cambio_porcentual) < 15:
                            st.write("\n- 🟡 **Cambio moderado:** Se espera una variación normal")
                        else:
                            st.write("\n- 🔴 **Cambio significativo:** Se prevé una alteración importante en la calidad del aire")
                        
                        st.write("""
                        **Factores que Influyen en la Predicción:**
                        """)
                        
                        # Analizar cada variable y su posible impacto
                        for interp in interpretaciones_corr:
                            var = interp['variable']
                            corr = interp['correlacion']
                            
                            if abs(corr) > 0.3:  # Solo variables con correlación significativa
                                valor_actual = data_24h[var].iloc[-1]
                                promedio_var = data_24h[var].mean()
                                desviacion_var = ((valor_actual - promedio_var) / promedio_var) * 100
                                
                                st.write(f"""
                                **{var}:**
                                - Correlación con PM2.5: {corr:.2f} ({'positiva' if corr > 0 else 'negativa'})
                                - Valor actual: {valor_actual:.1f} ({desviacion_var:+.1f}% vs promedio)
                                """)
                                
                                # Interpretación específica por variable
                                if var == 'PM10':
                                    if corr > 0.5:
                                        st.write("- **Impacto:** PM10 alto sugiere más material particulado en general")
                                elif var == 'Temperatura':
                                    if corr < -0.3:
                                        st.write("- **Impacto:** Mayor temperatura puede favorecer dispersión de contaminantes")
                                    elif corr > 0.3:
                                        st.write("- **Impacto:** Temperaturas altas pueden favorecer formación de partículas secundarias")
                                elif var == 'Humedad':
                                    if corr > 0.3:
                                        st.write("- **Impacto:** Alta humedad puede promover formación de aerosoles")
                                    elif corr < -0.3:
                                        st.write("- **Impacto:** Humedad favorece deposición húmeda de partículas")
                                elif var == 'NO2':
                                    if corr > 0.3:
                                        st.write("- **Impacto:** NO2 alto indica actividad de combustión que genera PM2.5")
                        
                        st.write("""
                        **Evaluación de Confiabilidad:**
                        """)
                        
                        if variables_estables >= 3:
                            st.write("\n- ✅ **Alta confiabilidad:** Mayoría de variables muestran patrones estables")
                        elif variables_estables >= 2:
                            st.write("\n- 🟡 **Confiabilidad moderada:** Algunas variables muestran variabilidad")
                        else:
                            st.write("\n- ⚠️ **Usar con cautela:** Alta variabilidad en las condiciones ambientales")
                        
                        if correlaciones_significativas >= 2:
                            st.write("\n- ✅ **Relaciones consistentes:** Múltiples variables correlacionadas facilitan predicción")
                        else:
                            st.write("\n- ⚠️ **Relaciones débiles:** Pocas correlaciones significativas detectadas")
                        
                        st.write("""
                        **Recomendaciones Basadas en la Predicción:**
                        """)
                        
                        calidad_actual = interpretar_calidad_aire(historical_pm25[-1])
                        calidad_predicha = interpretar_calidad_aire(final_pm25)
                        
                        if calidad_predicha['categoria'] != calidad_actual['categoria']:
                            st.write(f"""
                            - 🔄 **Cambio de categoría:** De "{calidad_actual['categoria']}" a "{calidad_predicha['categoria']}"
                            - 📋 **Nuevas recomendaciones:** {calidad_predicha['recomendaciones']}
                            """)
                        else:
                            st.write(f"""
                            - ✅ **Categoría estable:** Se mantiene en "{calidad_predicha['categoria']}"
                            - 📋 **Recomendaciones:** {calidad_predicha['recomendaciones']}
                            """)
                        
                        st.write("""
                        **Variables Clave a Monitorear:**
                        """)
                        
                        variables_criticas = sorted(interpretaciones_corr, key=lambda x: abs(x['correlacion']), reverse=True)[:3]
                        for i, var_info in enumerate(variables_criticas, 1):
                            st.write(f"""
                            {i}. **{var_info['variable']}** (correlación: {var_info['correlacion']:.2f})
                            """)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Comparación numérica con métricas mejoradas
                    st.subheader("📊 Comparación Numérica Detallada")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            label="PM2.5 Actual (última medición)",
                            value=f"{historical_pm25[-1]:.2f} µg/m³"
                        )
                        st.markdown(f"**Categoría:** {calidad_actual['categoria']}")
                    
                    with col2:
                        st.metric(
                            label="PM2.5 Predicho (+24h)",
                            value=f"{final_pm25:.2f} µg/m³",
                            delta=f"{cambio_absoluto:+.2f}"
                        )
                        st.markdown(f"**Categoría:** {calidad_predicha['categoria']}")
                    
                    with col3:
                        st.metric(
                            label="Cambio Porcentual",
                            value=f"{cambio_porcentual:+.1f}%"
                        )
                        tendencia = "↗️ Aumentando" if cambio_absoluto > 0 else "↘️ Disminuyendo" if cambio_absoluto < 0 else "➡️ Estable"
                        st.markdown(f"**Tendencia:** {tendencia}")
                    
                    with col4:
                        # Promedio de las últimas 24h para contexto
                        promedio_24h = np.mean(historical_pm25)
                        st.metric(
                            label="Promedio 24h",
                            value=f"{promedio_24h:.2f} µg/m³"
                        )
                        diferencia_promedio = ((final_pm25 - promedio_24h) / promedio_24h) * 100
                        st.markdown(f"**vs Promedio:** {diferencia_promedio:+.1f}%")
                    
                    # Tabla resumen de interpretación
                    st.subheader("📋 Resumen Ejecutivo")
                    
                    # Crear DataFrame para la tabla resumen
                    resumen_data = {
                        "Aspecto": [
                            "Calidad Aire Actual",
                            "Calidad Aire Predicha", 
                            "Tendencia de Cambio",
                            "Confiabilidad del Modelo",
                            "Variables Más Influyentes"
                        ],
                        "Valor/Estado": [
                            f"{calidad_actual['categoria']} ({historical_pm25[-1]:.1f} µg/m³)",
                            f"{calidad_predicha['categoria']} ({final_pm25:.1f} µg/m³)",
                            f"{cambio_porcentual:+.1f}% en 24h",
                            f"{variables_estables}/{len(variables)} variables estables",
                            ", ".join([v['variable'] for v in variables_criticas[:2]])
                        ],
                        "Interpretación": [
                            calidad_actual['descripcion'][:50] + "...",
                            calidad_predicha['descripcion'][:50] + "...",
                            "Estable" if abs(cambio_porcentual) < 10 else "Variable",
                            "Alta" if variables_estables >= 3 else "Media",
                            "Factores de mayor correlación identificados"
                        ]
                    }
                    
                    df_resumen = pd.DataFrame(resumen_data)
                    st.dataframe(df_resumen, use_container_width=True, hide_index=True)
                    
                    # Alertas y advertencias finales
                    if final_pm25 > 55.4:
                        st.markdown("""
                        <div class="danger-box">
                        <h4>⚠️ ALERTA: Calidad del Aire Insalubre</h4>
                        <p>La predicción indica niveles de PM2.5 que pueden ser perjudiciales para la salud. 
                        Se recomienda tomar precauciones especiales y limitar actividades al aire libre.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif final_pm25 > 35.4:
                        st.markdown("""
                        <div class="warning-box">
                        <h4>⚡ ATENCIÓN: Calidad del Aire Moderada a Insalubre</h4>
                        <p>Grupos sensibles (niños, adultos mayores, personas con problemas respiratorios) 
                        deben considerar limitar actividades prolongadas al aire libre.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif final_pm25 <= 12:
                        st.markdown("""
                        <div class="success-box">
                        <h4>✅ EXCELENTE: Calidad del Aire Buena</h4>
                        <p>Condiciones favorables para actividades al aire libre. 
                        La calidad del aire presenta poco o ningún riesgo.</p>
                        </div>
                        """, unsafe_allow_html=True)

    except Exception as e:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.error(f"❌ Error al procesar el archivo: {e}")