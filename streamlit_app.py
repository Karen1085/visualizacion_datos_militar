import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# 1. CONFIGURACIÓN DE INTERFAZ CORPORATIVA (ESTILO CYBERPUNK / NEON)
st.set_page_config(page_title="Dashboard Ley 1780 - Tesis Karen", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    header[data-testid="stHeader"] { visibility: hidden; display: none; }
    /* Fondo morado profundo */
    .stApp { background-color: #0d0614; font-family: 'Segoe UI', Tahoma, sans-serif; }
    /* Panel lateral */
    [data-testid="stSidebar"] { background-color: #130a1f !important; border-right: 1px solid #2a1642; }
    [data-testid="stSidebar"] * { color: #e0d4f5 !important; }
    /* Filtros fondo oscuro morado */
    div[data-baseweb="select"] > div { background-color: #1a0d2b !important; border: 1px solid #3d2063 !important; color: #ffffff !important; }
    ul[data-baseweb="menu"] { background-color: #1a0d2b !important; }
    ul[data-baseweb="menu"] li { color: #ffffff !important; }
    span[data-baseweb="tag"] { background-color: #00e5ff !important; color: #000000 !important; font-weight: bold;}
    /* Títulos y textos */
    h1, h2, h3, h4 { color: #ffffff !important; font-weight: 500; font-size: 1.1rem; margin-bottom: 0px; }
    p, label { color: #bbaacc !important; }
    /* Tarjetas de Indicadores (KPIs) estilo Neon */
    div[data-testid="metric-container"] { 
        background-color: #170a29; border-top: 3px solid #b400ff; 
        padding: 15px 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricLabel"] { font-size: 0.80rem !important; color: #bbaacc !important; text-transform: uppercase; }
    div[data-testid="stMetricValue"] > div { color: #00e5ff !important; font-size: 1.5rem !important; font-weight: bold; }
    .reportview-container .main .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 98%; }
</style>
""", unsafe_allow_html=True)

st.title("Impacto de la Ley 1780 (ProJoven) en el Mercado Laboral")
st.markdown("Análisis Econométrico Interactivo: Formalidad, Participación y Salarios (Evaluación de Impacto DiD)")
st.markdown("---")

# 2. CARGA Y PREPARACIÓN DE DATOS (Manejando el Parquet)
@st.cache_data
def load_data():
    df = pd.read_parquet("datos_tesis.parquet")
    
    # Asegurar formato fecha para el eje de tiempo
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # Crear dummies numéricas para el cálculo ponderado
    df['formal_num'] = np.where(df['formal_ss'].str.lower() == 'formal', 1, 0)
    df['part_num'] = np.where(df['part_mercadol'].str.lower() == 'participa', 1, 0)
    
    return df

df = load_data()

# 3. PANEL LATERAL DE SEGMENTACIÓN (Los filtros para Thomas)
st.sidebar.markdown("### 🔍 Filtros Econométricos")
clase_sel = st.sidebar.multiselect("Zona (Urbano/Rural)", options=sorted(df['clase'].dropna().unique()), default=df['clase'].dropna().unique())
area_sel = st.sidebar.multiselect("Ciudades Principales", options=sorted(df['area'].dropna().unique()), default=df['area'].dropna().unique()[:5]) # Por defecto 5 para no saturar
posicion_sel = st.sidebar.multiselect("Posición Ocupacional (Test Placebo)", options=sorted(df['posicion_ocup'].dropna().unique()), default=["Asalariados (Empresa/Gobierno)"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Parámetros de Tiempo")
ventana_sel = st.sidebar.radio("Ventana de Evento (Política: Mayo 2016)", 
                               ["Simetría Total (Nov 2012 - Feb 2020)", 
                                "24 meses antes / 12 meses después",
                                "12 meses antes / 12 meses después",
                                "6 meses antes / 12 meses después"])

suavizado = st.sidebar.slider("Meses de Suavizado (Media Móvil)", min_value=1, max_value=12, value=3)

# Aplicar filtros estáticos
df_filtered = df[(df['clase'].isin(clase_sel)) & 
                 (df['area'].isin(area_sel)) & 
                 (df['posicion_ocup'].isin(posicion_sel))]

# Lógica de Ventana de Tiempo (Event Study)
fecha_ley = pd.to_datetime('2016-05-01')
if "24 meses" in ventana_sel:
    df_filtered = df_filtered[(df_filtered['fecha'] >= fecha_ley - pd.DateOffset(months=24)) & (df_filtered['fecha'] <= fecha_ley + pd.DateOffset(months=12))]
elif "12 meses antes" in ventana_sel:
    df_filtered = df_filtered[(df_filtered['fecha'] >= fecha_ley - pd.DateOffset(months=12)) & (df_filtered['fecha'] <= fecha_ley + pd.DateOffset(months=12))]
elif "6 meses" in ventana_sel:
    df_filtered = df_filtered[(df_filtered['fecha'] >= fecha_ley - pd.DateOffset(months=6)) & (df_filtered['fecha'] <= fecha_ley + pd.DateOffset(months=12))]

# 4. MOTOR DE CÁLCULO PONDERADO (Lo que dejará mudo a Thomas)
def calcular_estadisticas_ponderadas(data):
    peso_total = data['fex18'].sum()
    if peso_total == 0: return pd.Series([0,0,0])
    
    tasa_formalidad = (data['formal_num'] * data['fex18']).sum() / peso_total
    tasa_participacion = (data['part_num'] * data['fex18']).sum() / peso_total
    # Salario promedio solo para ocupados con ingresos > 0
    df_salario = data[data['inglabo_real'] > 0]
    salario_promedio = (df_salario['inglabo_real'] * df_salario['fex18']).sum() / df_salario['fex18'].sum() if not df_salario.empty else 0
    
    return pd.Series([tasa_formalidad, tasa_participacion, salario_promedio], 
                     index=['Tasa_Formalidad', 'Tasa_Participacion', 'Salario_Real_Promedio'])

# Agrupar por mes y grupo de edad para las series de tiempo
ts_data = df_filtered.groupby(['fecha', 'young']).apply(calcular_estadisticas_ponderadas).reset_index()

# Aplicar suavizado y cálculos de cambios
for col in ['Tasa_Formalidad', 'Tasa_Participacion', 'Salario_Real_Promedio']:
    # 1. Promedio Suavizado (Niveles)
    ts_data[f'{col}_Suavizado'] = ts_data.groupby('young')[col].transform(lambda x: x.rolling(suavizado, min_periods=1).mean())
    # 2. Cambios Interanuales (t vs t-12) para quitar estacionalidad
    ts_data[f'{col}_Cambio_YoY'] = ts_data.groupby('young')[col].transform(lambda x: x.diff(periods=12))

# 5. INDICADORES MACRO (KPIs)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Muestra Filtrada (N)", f"{len(df_filtered):,}")
k2.metric("Población Expandida", f"{int(df_filtered['fex18'].sum()):,}")
tasa_form_global = (df_filtered['formal_num'] * df_filtered['fex18']).sum() / df_filtered['fex18'].sum()
k3.metric("Formalidad Promedio (Global)", f"{tasa_form_global*100:.1f}%")
tasa_part_global = (df_filtered['part_num'] * df_filtered['fex18']).sum() / df_filtered['fex18'].sum()
k4.metric("Participación Promedio (Global)", f"{tasa_part_global*100:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# 6. CONFIGURACIÓN GLOBAL DE GRÁFICAS (Plotly)
layout_config = dict(
    paper_bgcolor='#170a29', plot_bgcolor='#170a29', font=dict(color="#d4c5e8", size=12), margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(showgrid=False, zeroline=False, color='#bbaacc'),
    yaxis=dict(showgrid=True, gridcolor='#2a1642', zeroline=False, color='#bbaacc', tickformat=".1%"),
    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
)
colores_grupos = {'Hombres 18-24': '#00e5ff', 'Hombres 25-28': '#b400ff', 'Hombres 29-32': '#ff007f', 'Mujeres': '#39ff14'}

# ==========================================
# 7. GRÁFICAS DE SERIES DE TIEMPO
# ==========================================
c1, c2 = st.columns(2)

with c1:
    st.markdown("#### Tasa de Formalidad (Niveles Suavizados)")
    fig_form = px.line(ts_data, x='fecha', y='Tasa_Formalidad_Suavizado', color='young', color_discrete_map=colores_grupos)
    fig_form.add_vline(x=fecha_ley.timestamp() * 1000, line_dash="dash", line_color="#39ff14", annotation_text="Ley 1780", annotation_position="top right")
    fig_form.update_layout(**layout_config, height=400)
    st.plotly_chart(fig_form, use_container_width=True)

with c2:
    st.markdown("#### Cambios en la Formalidad (Variación Interanual)")
    # El cambio interanual no usa tickformat porcentual directo para evitar confusiones de escala
    fig_cambio = px.line(ts_data.dropna(subset=['Tasa_Formalidad_Cambio_YoY']), x='fecha', y='Tasa_Formalidad_Cambio_YoY', color='young', color_discrete_map=colores_grupos)
    fig_cambio.add_vline(x=fecha_ley.timestamp() * 1000, line_dash="dash", line_color="#39ff14")
    fig_cambio.update_layout(**layout_config, height=400)
    fig_cambio.update_layout(yaxis=dict(tickformat=".2f", title="Cambio Puntos Porcentuales (t vs t-12)"))
    st.plotly_chart(fig_cambio, use_container_width=True)

c3, c4 = st.columns(2)

with c3:
    st.markdown("#### Tasa de Participación Laboral (PEA)")
    fig_part = px.line(ts_data, x='fecha', y='Tasa_Participacion_Suavizado', color='young', color_discrete_map=colores_grupos)
    fig_part.add_vline(x=fecha_ley.timestamp() * 1000, line_dash="dash", line_color="#ff8c00")
    fig_part.update_layout(**layout_config, height=400)
    st.plotly_chart(fig_part, use_container_width=True)

with c4:
    st.markdown("#### Salario Real Promedio (Ocupados)")
    fig_sal = px.line(ts_data, x='fecha', y='Salario_Real_Promedio_Suavizado', color='young', color_discrete_map=colores_grupos)
    fig_sal.add_vline(x=fecha_ley.timestamp() * 1000, line_dash="dash", line_color="#00e5ff")
    fig_sal.update_layout(**layout_config, height=400)
    fig_sal.update_layout(yaxis=dict(tickformat="$,.0f"))
    st.plotly_chart(fig_sal, use_container_width=True)

st.markdown("---")

# ==========================================
# 8. ESTADÍSTICAS DESCRIPTIVAS PONDERADAS (Para Thomas)
# ==========================================
st.markdown("#### Estadísticas Descriptivas de Controles Ponderados (Pre-Política vs Post-Política)")

# Crear variable dummy de Post-Política
df_filtered['Post_Politica'] = np.where(df_filtered['fecha'] >= fecha_ley, 'Post (>= Mayo 2016)', 'Pre (< Mayo 2016)')

def calcular_controles(data):
    peso = data['fex18'].sum()
    if peso == 0: return pd.Series([0,0])
    edad_prom = (data['edad'] * data['fex18']).sum() / peso
    escolaridad_prom = (data['años_escolaridad'] * data['fex18']).sum() / peso
    return pd.Series([edad_prom, escolaridad_prom], index=['Edad Promedio', 'Años de Escolaridad'])

# Tabla pivote agrupada
tabla_desc = df_filtered.groupby(['young', 'Post_Politica']).apply(calcular_controles).reset_index()
# Redondear para estética
tabla_desc['Edad Promedio'] = tabla_desc['Edad Promedio'].round(1)
tabla_desc['Años de Escolaridad'] = tabla_desc['Años de Escolaridad'].round(1)

# Mostrar DataFrame con diseño nativo de Streamlit
st.dataframe(tabla_desc, use_container_width=True, hide_index=True)