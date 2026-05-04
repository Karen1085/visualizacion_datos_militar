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
    [data-testid="stSidebar"] * { color: #e0d4f5 !important; }
    div[data-baseweb="select"] > div { background-color: #1a0d2b !important; border: 1px solid #3d2063 !important; color: #ffffff !important; }
    ul[data-baseweb="menu"] { background-color: #1a0d2b !important; }
    ul[data-baseweb="menu"] li { color: #ffffff !important; }
    span[data-baseweb="tag"] { background-color: #00e5ff !important; color: #000000 !important; font-weight: bold;}
    h1, h2, h3, h4 { color: #ffffff !important; font-weight: 500; font-size: 1.1rem; margin-bottom: 0px; }
    p, label { color: #bbaacc !important; }
    div[data-testid="metric-container"] { 
        background-color: #170a29; border-top: 3px solid #b400ff; 
        padding: 15px 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricLabel"] { font-size: 0.80rem !important; color: #bbaacc !important; text-transform: uppercase; }
    div[data-testid="stMetricValue"] > div { color: #00e5ff !important; font-size: 1.5rem !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("Impacto de la Ley 1780 (ProJoven) en el Mercado Laboral")
st.markdown("Análisis de Formalidad, Participación y Salarios con Factor de Expansión (GEIH)")
st.markdown("---")

# 2. CARGA Y PREPARACIÓN DE DATOS
@st.cache_data
def load_data():
    # Cargar el parquet generado
    df = pd.read_parquet("datos_tesis.parquet")
    
    # Asegurar que fecha sea datetime
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # Convertir variables objetivo a dummies numéricas para cálculos ponderados
    # Basado en tu muestra: formal_ss ('formal'/'informal') y part_mercadol ('Participa'/'Inactivo')
    df['formal_num'] = np.where(df['formal_ss'].str.lower() == 'formal', 1.0, 0.0)
    df['part_num'] = np.where(df['part_mercadol'].str.lower().str.contains('participa'), 1.0, 0.0)
    
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error al cargar 'datos_tesis.parquet': {e}")
    st.stop()

# 3. PANEL LATERAL (FILTROS THOMAS)
st.sidebar.markdown("### 🔍 Segmentación de Muestra")
clase_sel = st.sidebar.multiselect("Clase (Urbano/Rural)", options=df['clase'].unique(), default=df['clase'].unique())
area_sel = st.sidebar.multiselect("Áreas (Ciudades)", options=sorted(df['area'].unique()), default=df['area'].unique()[:5])
posicion_sel = st.sidebar.multiselect("Posición Ocupacional", options=df['posicion_ocup'].unique(), default=["Asalariados (Empresa/Gobierno)"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Configuración de Tendencias")
ventana_sel = st.sidebar.radio("Ventana de Tiempo (Ley: Mayo 2016)", 
                               ["Simetría Total", "24m antes / 12m después", "12m antes / 12m después", "6m antes / 12m después"])

suavizado = st.sidebar.slider("Meses de Suavizado (Media Móvil)", 1, 12, 3)

# Aplicar filtros
df_f = df[(df['clase'].isin(clase_sel)) & (df['area'].isin(area_sel)) & (df['posicion_ocup'].isin(posicion_sel))].copy()

# Lógica de Ventanas de Tiempo
fecha_ley = pd.to_datetime('2016-05-01')
if "24m" in ventana_sel:
    df_f = df_f[(df_f['fecha'] >= fecha_ley - pd.DateOffset(months=24)) & (df_f['fecha'] <= fecha_ley + pd.DateOffset(months=12))]
elif "12m" in ventana_sel:
    df_f = df_f[(df_f['fecha'] >= fecha_ley - pd.DateOffset(months=12)) & (df_f['fecha'] <= fecha_ley + pd.DateOffset(months=12))]
elif "6m" in ventana_sel:
    df_f = df_f[(df_f['fecha'] >= fecha_ley - pd.DateOffset(months=6)) & (df_f['fecha'] <= fecha_ley + pd.DateOffset(months=12))]

# 4. MOTOR DE CÁLCULO PONDERADO (REPARADO PARA EVITAR KEYERROR)
def weighted_stats(data):
    w = data['fex18'].sum()
    cols = ['Tasa_Formalidad', 'Tasa_Participacion', 'Salario_Real']
    if w == 0:
        return pd.Series([0.0, 0.0, 0.0], index=cols)
    
    f = (data['formal_num'] * data['fex18']).sum() / w
    p = (data['part_num'] * data['fex18']).sum() / w
    
    # Salario solo para ocupados
    df_s = data[data['inglabo_real'] > 0]
    s = (df_s['inglabo_real'] * df_s['fex18']).sum() / df_s['fex18'].sum() if df_s['fex18'].sum() > 0 else 0.0
    
    return pd.Series([f, p, s], index=cols)

# Agrupación por mes y grupo de edad (young)
if not df_f.empty:
    ts_data = df_f.groupby(['fecha', 'young']).apply(weighted_stats, include_groups=False).reset_index()
    
    # Cálculos de Suavizado y Cambios YoY
    for target in ['Tasa_Formalidad', 'Tasa_Participacion', 'Salario_Real']:
        ts_data[f'{target}_S'] = ts_data.groupby('young')[target].transform(lambda x: x.rolling(suavizado, min_periods=1).mean())
        ts_data[f'{target}_Diff'] = ts_data.groupby('young')[target].transform(lambda x: x.diff(12))
else:
    st.warning("No hay datos para los filtros seleccionados.")
    st.stop()

# 5. KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Población Expandida", f"{int(df_f['fex18'].sum()):,}")
k2.metric("Muestra (N)", f"{len(df_f):,}")
k3.metric("Formalidad Promedio", f"{ts_data['Tasa_Formalidad'].mean()*100:.1f}%")
k4.metric("Participación Promedio", f"{ts_data['Tasa_Participacion'].mean()*100:.1f}%")

# 6. GRÁFICAS
layout_ui = dict(
    paper_bgcolor='#170a29', plot_bgcolor='#170a29', font=dict(color="#d4c5e8"),
    xaxis=dict(showgrid=False, color='#bbaacc'),
    yaxis=dict(showgrid=True, gridcolor='#2a1642', color='#bbaacc'),
    legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center")
)
colores = {'Hombres 18-24': '#00e5ff', 'Hombres 25-28': '#b400ff', 'Hombres 29-32': '#ff007f', 'Mujeres': '#39ff14'}

c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Tasa de Formalidad (Niveles Suavizados)")
    fig1 = px.line(ts_data, x='fecha', y='Tasa_Formalidad_S', color='young', color_discrete_map=colores)
    fig1.add_vline(x=fecha_ley.timestamp()*1000, line_dash="dash", line_color="#39ff14")
    fig1.update_layout(**layout_ui, yaxis=dict(tickformat=".1%"))
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.markdown("#### Cambios YoY en Formalidad (t vs t-12)")
    fig2 = px.line(ts_data.dropna(), x='fecha', y='Tasa_Formalidad_Diff', color='young', color_discrete_map=colores)
    fig2.add_vline(x=fecha_ley.timestamp()*1000, line_dash="dash", line_color="#39ff14")
    fig2.update_layout(**layout_ui)
    st.plotly_chart(fig2, use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.markdown("#### Tasa de Participación (PEA)")
    fig3 = px.line(ts_data, x='fecha', y='Tasa_Participacion_S', color='young', color_discrete_map=colores)
    fig3.update_layout(**layout_ui, yaxis=dict(tickformat=".1%"))
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    st.markdown("#### Salario Real Promedio")
    fig4 = px.line(ts_data, x='fecha', y='Salario_Real_S', color='young', color_discrete_map=colores)
    fig4.update_layout(**layout_ui, yaxis=dict(tickformat="$,.0f"))
    st.plotly_chart(fig4, use_container_width=True)

# 7. ESTADÍSTICAS DE CONTROLES (PARA THOMAS)
st.markdown("---")
st.markdown("#### Estadísticas Descriptivas de Controles (Ponderadas por FEX)")
df_f['Periodo'] = np.where(df_f['fecha'] < fecha_ley, 'Pre-Ley', 'Post-Ley')

def get_controls(x):
    w = x['fex18'].sum()
    if w == 0: return pd.Series([0,0], index=['Escolaridad', 'Edad'])
    esc = (x['años_escolaridad'] * x['fex18']).sum() / w
    ed = (x['edad'] * x['fex18']).sum() / w
    return pd.Series([esc, ed], index=['Escolaridad', 'Edad'])

desc_table = df_f.groupby(['young', 'Periodo']).apply(get_controls, include_groups=False).reset_index()
st.dataframe(desc_table.sort_values(['young', 'Periodo'], ascending=[True, False]), use_container_width=True, hide_index=True)
