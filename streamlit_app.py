import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# 1. CONFIGURACIÓN DE INTERFAZ CORPORATIVA (ESTILO CYBERPUNK / NEON)
st.set_page_config(page_title="Tesis Karen - Impacto Ley 1780", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    header[data-testid="stHeader"] { visibility: hidden; display: none; }
    .stApp { background-color: #0d0614; font-family: 'Segoe UI', Tahoma, sans-serif; }
    [data-testid="stSidebar"] { background-color: #130a1f !important; border-right: 1px solid #2a1642; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    div[data-baseweb="select"] > div { background-color: #1a0d2b !important; border: 1px solid #3d2063 !important; color: #ffffff !important; }
    ul[data-baseweb="menu"] { background-color: #1a0d2b !important; }
    ul[data-baseweb="menu"] li { color: #ffffff !important; }
    h1, h2, h3, h4 { color: #ffffff !important; font-weight: 500; }
    p, label { color: #ffffff !important; }
    div[data-testid="metric-container"] { 
        background-color: #170a29; border-top: 3px solid #b400ff; 
        padding: 15px 15px; border-radius: 8px;
    }
    div[data-testid="stMetricLabel"] { color: #ffffff !important; text-transform: uppercase; }
    div[data-testid="stMetricValue"] > div { color: #00e5ff !important; font-size: 1.8rem !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("Impacto de la Ley 1780 (ProJoven) en el Mercado Laboral")
st.markdown("Análisis de Formalidad, Participación (PEA) y Salarios - Factor de Expansión GEIH")
st.markdown("---")

# 2. CARGA Y PREPARACIÓN DE DATOS
@st.cache_data
def load_data():
    df = pd.read_parquet("datos_tesis.parquet")
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # LIMPIEZA ROBUSTA DE VARIABLES (Para evitar líneas planas en 100%)
    df['formal_num'] = np.where(df['formal_ss'].astype(str).str.strip().str.lower() == 'formal', 1.0, 0.0)
    # Buscamos 'participa' para que asigne 1, si es 'no participa' o 'inactivo' asigne 0
    df['part_num'] = np.where(df['part_mercadol'].astype(str).str.strip().str.lower() == 'participa', 1.0, 0.0)
    
    return df

df = load_data()

# 3. PANEL LATERAL (FILTROS)
st.sidebar.markdown("### Filtros de Análisis")
clase_options = sorted(df['clase'].dropna().astype(str).unique())
area_options = sorted(df['area'].dropna().astype(str).unique())
posicion_options = sorted(df['posicion_ocup'].dropna().astype(str).unique())

clase_sel = st.sidebar.multiselect("Zona", options=clase_options, default=clase_options)
area_sel = st.sidebar.multiselect("Áreas Metropolitano", options=area_options, default=area_options[:5])
posicion_sel = st.sidebar.multiselect("Posición Ocupacional", options=posicion_options, default=["Asalariados (Empresa/Gobierno)"])

ventana_sel = st.sidebar.radio("Ventana de Tiempo", ["Simetría Total", "24m antes / 12m después", "12m antes / 12m después"])
suavizado = st.sidebar.slider("Suavizado (Meses)", 1, 12, 3)

# Aplicar filtros
df_f = df[(df['clase'].astype(str).isin(clase_sel)) & 
          (df['area'].astype(str).isin(area_sel)) & 
          (df['posicion_ocup'].astype(str).isin(posicion_sel))].copy()

# 4. MOTOR DE CÁLCULO PONDERADO
def weighted_stats(data):
    w = data['fex18'].sum()
    cols = ['Tasa_Formalidad', 'Tasa_Participacion', 'Salario_Real']
    if w == 0: return pd.Series([0.0, 0.0, 0.0], index=cols)
    
    f = (data['formal_num'] * data['fex18']).sum() / w
    p = (data['part_num'] * data['fex18']).sum() / w
    
    df_s = data[data['inglabo_real'] > 0]
    s = (df_s['inglabo_real'] * df_s['fex18']).sum() / df_s['fex18'].sum() if df_s['fex18'].sum() > 0 else 0.0
    return pd.Series([f, p, s], index=cols)

# Agrupación por mes y grupo
if not df_f.empty:
    ts_data = df_f.groupby(['fecha', 'young']).apply(weighted_stats, include_groups=False).reset_index()
    for target in ['Tasa_Formalidad', 'Tasa_Participacion', 'Salario_Real']:
        ts_data[f'{target}_S'] = ts_data.groupby('young')[target].transform(lambda x: x.rolling(suavizado, min_periods=1).mean())
        ts_data[f'{target}_Diff'] = ts_data.groupby('young')[target].transform(lambda x: x.diff(12))
else:
    st.stop()

# 5. CONFIGURACIÓN DE GRÁFICAS (ETIQUETAS EN BLANCO)
# Actualicé los colores a blanco (#ffffff) para máxima visibilidad
layout_ui = dict(
    paper_bgcolor='#170a29', plot_bgcolor='#170a29', 
    font=dict(color="#ffffff", size=12), # Texto general en blanco
    xaxis=dict(showgrid=False, color='#ffffff'),
    yaxis=dict(showgrid=True, gridcolor='#2a1642', color='#ffffff'),
    legend=dict(
        font=dict(color="#ffffff"), # Nombres de grupos en blanco
        orientation="h", y=-0.2, x=0.5, xanchor="center"
    )
)
colores_filt = {'Hombres 18-24': '#00e5ff', 'Hombres 25-28': '#b400ff', 'Hombres 29-32': '#ff007f', 'Mujeres': '#39ff14'}

# 6. VISUALIZACIÓN
c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Tasa de Formalidad (Niveles)")
    fig1 = px.line(ts_data, x='fecha', y='Tasa_Formalidad_S', color='young', color_discrete_map=colores_filt)
    fig1.update_layout(**layout_ui)
    fig1.update_yaxes(tickformat=".1%", range=[0, 1])
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.markdown("#### Tasa de Participación (PEA)")
    fig2 = px.line(ts_data, x='fecha', y='Tasa_Participacion_S', color='young', color_discrete_map=colores_filt)
    fig2.update_layout(**layout_ui)
    # Ajustamos el rango de 0 a 1 (0% a 100%) para que no se vea plana en el techo
    fig2.update_yaxes(tickformat=".1%", range=[0, 1]) 
    st.plotly_chart(fig2, use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.markdown("#### Cambios YoY Formalidad (t vs t-12)")
    fig3 = px.line(ts_data.dropna(), x='fecha', y='Tasa_Formalidad_Diff', color='young', color_discrete_map=colores_filt)
    fig3.update_layout(**layout_ui)
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    st.markdown("#### Salario Real Promedio")
    fig4 = px.line(ts_data, x='fecha', y='Salario_Real_S', color='young', color_discrete_map=colores_filt)
    fig4.update_layout(**layout_ui)
    fig4.update_yaxes(tickformat="$,.0f")
    st.plotly_chart(fig4, use_container_width=True)

# 7. TABLA DE CONTROLES
st.markdown("---")
st.markdown("#### Resumen Descriptivo de Controles Ponderados")
df_f['Periodo'] = np.where(df_f['fecha'] < pd.to_datetime('2016-05-01'), 'Pre-Ley', 'Post-Ley')
def get_c(x):
    w = x['fex18'].sum()
    if w == 0: return pd.Series([0,0], index=['Escolaridad', 'Edad'])
    return pd.Series([(x['años_escolaridad']*x['fex18']).sum()/w, (x['edad']*x['fex18']).sum()/w], index=['Escolaridad', 'Edad'])

desc = df_f.groupby(['young', 'Periodo']).apply(get_c, include_groups=False).reset_index()
st.dataframe(desc, use_container_width=True, hide_index=True)
