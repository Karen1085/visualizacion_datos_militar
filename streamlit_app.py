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
    try:
        df = pd.read_parquet("datos_tesis.parquet")
    except:
        st.error("Archivo 'datos_tesis.parquet' no encontrado.")
        st.stop()
        
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # Limpieza de textos para evitar errores de coincidencia
    df['formal_num'] = np.where(df['formal_ss'].astype(str).str.strip().str.lower() == 'formal', 1.0, 0.0)
    df['part_num'] = np.where(df['part_mercadol'].astype(str).str.strip().str.lower() == 'participa', 1.0, 0.0)
    
    return df

df = load_data()

# 3. PANEL LATERAL (FILTROS)
st.sidebar.markdown("### 🔍 Filtros Territoriales")
clase_options = sorted(df['clase'].dropna().astype(str).unique())
area_options = sorted(df['area'].dropna().astype(str).unique())
posicion_options = sorted(df['posicion_ocup'].dropna().astype(str).unique())

clase_sel = st.sidebar.multiselect("Zona (Clase)", options=clase_options, default=clase_options)
area_sel = st.sidebar.multiselect("Áreas Metropolitanas", options=area_options, default=area_options[:5])

st.sidebar.markdown("### 💼 Filtro de Ocupación (Solo Formalidad/Salarios)")
default_pos = ["Asalariados (Empresa/Gobierno)"] if "Asalariados (Empresa/Gobierno)" in posicion_options else posicion_options[:1]
posicion_sel = st.sidebar.multiselect("Posición Ocupacional", options=posicion_options, default=default_pos)

st.sidebar.markdown("---")
ventana_sel = st.sidebar.radio("Ventana de Tiempo", ["Simetría Total", "24m antes / 12m después", "12m antes / 12m después"])
suavizado = st.sidebar.slider("Suavizado (Meses)", 1, 12, 3)

# 4. LÓGICA DE FILTRADO DOBLE (Para evitar líneas planas en participación)
# df_geo: Solo filtros territoriales (Se usa para calcular PARTICIPACIÓN / PEA)
df_geo = df[(df['clase'].astype(str).isin(clase_sel)) & (df['area'].astype(str).isin(area_sel))].copy()

# df_f: Filtros territoriales + Ocupación (Se usa para FORMALIDAD y SALARIOS)
df_f = df_geo[df_geo['posicion_ocup'].astype(str).isin(posicion_sel)].copy()

# Aplicar ventana de tiempo a ambos
fecha_ley = pd.to_datetime('2016-05-01')
def aplicar_ventana(data, ventana):
    if "24m" in ventana:
        return data[(data['fecha'] >= fecha_ley - pd.DateOffset(months=24)) & (data['fecha'] <= fecha_ley + pd.DateOffset(months=12))]
    elif "12m" in ventana:
        return data[(data['fecha'] >= fecha_ley - pd.DateOffset(months=12)) & (data['fecha'] <= fecha_ley + pd.DateOffset(months=12))]
    return data

df_geo = aplicar_ventana(df_geo, ventana_sel)
df_f = aplicar_ventana(df_f, ventana_sel)

# 5. MOTOR DE CÁLCULO PONDERADO
def calc_formal_salario(x):
    w = x['fex18'].sum()
    if w == 0: return pd.Series([0.0, 0.0], index=['Tasa_Formalidad', 'Salario_Real'])
    f = (x['formal_num'] * x['fex18']).sum() / w
    df_s = x[x['inglabo_real'] > 0]
    s = (df_s['inglabo_real'] * df_s['fex18']).sum() / df_s['fex18'].sum() if df_s['fex18'].sum() > 0 else 0.0
    return pd.Series([f, s], index=['Tasa_Formalidad', 'Salario_Real'])

def calc_participacion(x):
    w = x['fex18'].sum()
    if w == 0: return pd.Series([0.0], index=['Tasa_Participacion'])
    p = (x['part_num'] * x['fex18']).sum() / w
    return pd.Series([p], index=['Tasa_Participacion'])

# Agrupaciones
if not df_f.empty:
    # Formalidad y Salario (Dependen de Ocupación)
    ts_form = df_f.groupby(['fecha', 'young']).apply(calc_formal_salario, include_groups=False).reset_index()
    # Participación (Independiente de Ocupación para captar inactivos)
    ts_part = df_geo.groupby(['fecha', 'young']).apply(calc_participacion, include_groups=False).reset_index()
    
    # Unir resultados
    ts_data = pd.merge(ts_form, ts_part, on=['fecha', 'young'])
    
    # Suavizado y Diferencias
    for target in ['Tasa_Formalidad', 'Tasa_Participacion', 'Salario_Real']:
        ts_data[f'{target}_S'] = ts_data.groupby('young')[target].transform(lambda x: x.rolling(suavizado, min_periods=1).mean())
        if target != 'Tasa_Participacion':
            ts_data[f'{target}_Diff'] = ts_data.groupby('young')[target].transform(lambda x: x.diff(12))
else:
    st.warning("No hay datos para estos filtros.")
    st.stop()

# 6. CONFIGURACIÓN VISUAL
layout_ui = dict(
    paper_bgcolor='#170a29', plot_bgcolor='#170a29', 
    font=dict(color="#ffffff", size=12),
    xaxis=dict(showgrid=False, color='#ffffff'),
    yaxis=dict(showgrid=True, gridcolor='#2a1642', color='#ffffff'),
    legend=dict(font=dict(color="#ffffff"), orientation="h", y=-0.2, x=0.5, xanchor="center")
)
colores = {'Hombres 18-24': '#00e5ff', 'Hombres 25-28': '#b400ff', 'Hombres 29-32': '#ff007f', 'Mujeres': '#39ff14'}

# 7. DASHBOARD
k1, k2, k3, k4 = st.columns(4)
k1.metric("Pob. Expandida (Ocupada)", f"{int(df_f['fex18'].sum()):,}")
k2.metric("Muestra (N)", f"{len(df_f):,}")
k3.metric("Formalidad Promedio", f"{ts_data['Tasa_Formalidad'].mean()*100:.1f}%")
k4.metric("Participación (PEA)", f"{ts_data['Tasa_Participacion'].mean()*100:.1f}%")

c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Tasa de Formalidad (Efecto Contrato)")
    fig1 = px.line(ts_data, x='fecha', y='Tasa_Formalidad_S', color='young', color_discrete_map=colores)
    fig1.add_vline(x=fecha_ley.timestamp()*1000, line_dash="dash", line_color="#39ff14", annotation_text="Ley 1780")
    fig1.update_layout(**layout_ui)
    fig1.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.markdown("#### Tasa de Participación (PET)")
    fig2 = px.line(ts_data, x='fecha', y='Tasa_Participacion_S', color='young', color_discrete_map=colores)
    fig2.update_layout(**layout_ui)
    fig2.update_yaxes(tickformat=".1%", range=[ts_data['Tasa_Participacion_S'].min()*0.9, 1.0])
    st.plotly_chart(fig2, use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.markdown("#### Cambios YoY Formalidad (t vs t-12)")
    fig3 = px.line(ts_data.dropna(subset=['Tasa_Formalidad_Diff']), x='fecha', y='Tasa_Formalidad_Diff', color='young', color_discrete_map=colores)
    fig3.update_layout(**layout_ui)
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    st.markdown("#### Salario Real Promedio")
    fig4 = px.line(ts_data, x='fecha', y='Salario_Real_S', color='young', color_discrete_map=colores)
    fig4.update_layout(**layout_ui)
    fig4.update_yaxes(tickformat="$,.0f")
    st.plotly_chart(fig4, use_container_width=True)

# 8. TABLA DE CONTROLES
st.markdown("---")
st.markdown("#### Resumen de Controles Ponderados")
df_f['Periodo'] = np.where(df_f['fecha'] < fecha_ley, 'Pre-Ley', 'Post-Ley')
def get_c(x):
    w = x['fex18'].sum()
    if w == 0: return pd.Series([0,0], index=['Escolaridad', 'Edad'])
    return pd.Series([(x['años_escolaridad']*x['fex18']).sum()/w, (x['edad']*x['fex18']).sum()/w], index=['Escolaridad', 'Edad'])

desc = df_f.groupby(['young', 'Periodo']).apply(get_c, include_groups=False).reset_index()
st.dataframe(desc.sort_values(['young', 'Periodo'], ascending=[True, False]), use_container_width=True, hide_index=True)
