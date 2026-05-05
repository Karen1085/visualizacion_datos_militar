import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# 1. CONFIGURACIÓN DE INTERFAZ (ESTILO NEON / CYBERPUNK)
st.set_page_config(page_title="Tesis Karen - Evaluación DiD Ley 1780", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    header[data-testid="stHeader"] { visibility: hidden; display: none; }
    .stApp { background-color: #0d0614; font-family: 'Segoe UI', Tahoma, sans-serif; }
    [data-testid="stSidebar"] { background-color: #130a1f !important; border-right: 1px solid #2a1642; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    div[data-baseweb="select"] > div { background-color: #1a0d2b !important; border: 1px solid #3d2063 !important; color: #ffffff !important; }
    ul[data-baseweb="menu"] { background-color: #1a0d2b !important; }
    ul[data-baseweb="menu"] li { color: #ffffff !important; }
    h1, h2, h3, h4 { color: #ffffff !important; font-weight: 500; font-size: 1.1rem; }
    p, label { color: #ffffff !important; }
    div[data-testid="metric-container"] { 
        background-color: #170a29; border-top: 3px solid #b400ff; 
        padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricLabel"] { color: #ffffff !important; text-transform: uppercase; font-size: 0.75rem !important; }
    div[data-testid="stMetricValue"] > div { color: #00e5ff !important; font-size: 1.7rem !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# 2. CARGA DE DATOS
@st.cache_data
def load_data():
    df = pd.read_parquet("datos_tesis.parquet")
    df['fecha'] = pd.to_datetime(df['fecha'])
    # Normalización de dummies numéricas
    df['formal_num'] = np.where(df['formal_ss'].astype(str).str.strip().str.lower() == 'formal', 1.0, 0.0)
    df['part_num'] = np.where(df['part_mercadol'].astype(str).str.strip().str.lower() == 'participa', 1.0, 0.0)
    return df

df = load_data()

# 3. PANEL LATERAL: TODOS LOS FILTROS
st.sidebar.markdown("### 📊 Variables de Control")

# Filtros Territoriales
clase_options = sorted(df['clase'].dropna().astype(str).unique())
area_options = sorted(df['area'].dropna().astype(str).unique())
dpto_options = sorted(df['dpto'].dropna().astype(str).unique())

clase_sel = st.sidebar.multiselect("Zona (Urbano/Rural)", clase_options, default=clase_options)
area_sel = st.sidebar.multiselect("Ciudades Principales", area_options, default=area_options[:3])
dpto_sel = st.sidebar.multiselect("Departamentos", dpto_options)

# Filtros Socioeconómicos
sexo_sel = st.sidebar.multiselect("Sexo", options=df['sexo'].unique(), default=df['sexo'].unique())
rol_sel = st.sidebar.multiselect("Rol en Hogar", options=df['rol_hogar'].unique(), default=df['rol_hogar'].unique())
est_sel = st.sidebar.multiselect("Estado Civil", options=df['estado_civil'].unique(), default=df['estado_civil'].unique())
pos_sel = st.sidebar.multiselect("Posición Ocupacional", options=df['posicion_ocup'].unique(), default=["Asalariados (Empresa/Gobierno)"])

st.sidebar.markdown("---")
ventana_sel = st.sidebar.radio("Ventana de Tiempo (Política: Mayo 2016)", 
                               ["1. 6m antes / 6m después", "2. 12m antes / 12m después", "3. 24m antes / 24m después", "4. Periodo Completo"])
suavizado = st.sidebar.slider("Meses de Suavizado", 1, 12, 3)

# 4. LÓGICA DE FILTRADO Y VENTANAS
fecha_ley = pd.to_datetime('2016-05-01')
df_f = df[(df['clase'].astype(str).isin(clase_sel)) & 
          (df['area'].astype(str).isin(area_sel)) &
          (df['sexo'].isin(sexo_sel)) &
          (df['rol_hogar'].isin(rol_sel)) &
          (df['estado_civil'].isin(est_sel)) &
          (df['posicion_ocup'].isin(pos_sel))].copy()

if dpto_sel:
    df_f = df_f[df_f['dpto'].isin(dpto_sel)]

def filtrar_ventana(data, ventana):
    if "1." in ventana: m = 6
    elif "2." in ventana: m = 12
    elif "3." in ventana: m = 24
    else: return data
    return data[(data['fecha'] >= fecha_ley - pd.DateOffset(months=m)) & (data['fecha'] <= fecha_ley + pd.DateOffset(months=m))]

df_f = filtrar_ventana(df_f, ventana_sel)

# 5. MOTOR DE CÁLCULO PONDERADO (FEX18)
def get_weighted_metrics(x):
    w = x['fex18'].sum()
    if w == 0: return pd.Series([0,0,0], index=['Formalidad', 'Participacion', 'Salario'])
    form = (x['formal_num'] * x['fex18']).sum() / w
    part = (x['part_num'] * x['fex18']).sum() / w
    df_s = x[x['inglabo_real'] > 0]
    sal = (df_s['inglabo_real'] * df_s['fex18']).sum() / df_s['fex18'].sum() if df_s['fex18'].sum() > 0 else 0
    return pd.Series([form, part, sal], index=['Formalidad', 'Participacion', 'Salario'])

# Procesamiento de Series
if not df_f.empty:
    ts = df_f.groupby(['fecha', 'young']).apply(get_weighted_metrics).reset_index()
    for col in ['Formalidad', 'Participacion', 'Salario']:
        ts[f'{col}_S'] = ts.groupby('young')[col].transform(lambda x: x.rolling(suavizado, min_periods=1).mean())
        ts[f'{target_col := col}_Diff'] = ts.groupby('young')[col].transform(lambda x: x.diff(12))
        
    # Pivot para Brechas DiD
    def calc_gap(data, metric):
        pivot = data.pivot(index='fecha', columns='young', values=f'{metric}_S')
        if 'Hombres 18-24' in pivot.columns and 'Hombres 25-28' in pivot.columns:
            return (pivot['Hombres 18-24'] - pivot['Hombres 25-28']).reset_index(name='Gap')
        return pd.DataFrame(columns=['fecha', 'Gap'])

else:
    st.error("Filtros sin datos.")
    st.stop()

# 6. DASHBOARD PRINCIPAL
st.title("Impacto Ley 1780: Evaluación de Resultados")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Población Expandida", f"{int(df_f['fex18'].sum()):,}")
sal_global = (df_f[df_f['inglabo_real']>0]['inglabo_real'] * df_f[df_f['inglabo_real']>0]['fex18']).sum() / df_f[df_f['inglabo_real']>0]['fex18'].sum()
k2.metric("Ingreso Laboral Promedio", f"${sal_global:,.0f}")
k3.metric("Tasa Formalidad", f"{(ts['Formalidad'].mean()*100):.1f}%")
k4.metric("Participación (PEA)", f"{(ts['Participacion'].mean()*100):.1f}%")

layout_ui = dict(paper_bgcolor='#170a29', plot_bgcolor='#170a29', font=dict(color="#ffffff", size=10),
                 margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
                 xaxis=dict(showgrid=False), yaxis=dict(gridcolor='#2a1642'))
colores = {'Hombres 18-24': '#00e5ff', 'Hombres 25-28': '#b400ff', 'Hombres 29-32': '#ff007f', 'Mujeres': '#39ff14'}

def render_row(metric_name, title, is_pct=True):
    st.markdown(f"### {title}")
    c1, c2, c3 = st.columns(3)
    with c1:
        fig = px.line(ts, x='fecha', y=f'{metric_name}_S', color='young', color_discrete_map=colores, title="Niveles Suavizados")
        fig.add_vline(x=fecha_ley.timestamp()*1000, line_dash="dash", line_color="#39ff14")
        fig.update_layout(**layout_ui)
        if is_pct: fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.line(ts.dropna(), x='fecha', y=f'{metric_name}_Diff', color='young', color_discrete_map=colores, title="Cambios YoY (t vs t-12)")
        fig.add_vline(x=fecha_ley.timestamp()*1000, line_dash="dash", line_color="#39ff14")
        fig.update_layout(**layout_ui)
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        gap_df = calc_gap(ts, metric_name)
        fig = px.bar(gap_df, x='fecha', y='Gap', title="Validación DiD (Brecha T - C)", color='Gap', color_continuous_scale=['#ff007f', '#00e5ff'])
        fig.add_vline(x=fecha_ley.timestamp()*1000, line_dash="dash", line_color="#39ff14")
        fig.update_layout(**layout_ui, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

# Renderizado de Filas Objetivo
render_row('Formalidad', "1. Análisis de Formalidad")
render_row('Salario', "2. Análisis de Salarios Reales", is_pct=False)
render_row('Participacion', "3. Análisis de Participación Laboral (PEA)")

# 7. TABLA DE CONTROLES
st.markdown("---")
st.markdown("#### Estadísticas Descriptivas de Controles Ponderados")
df_f['Periodo'] = np.where(df_f['fecha'] < fecha_ley, 'Pre-Ley', 'Post-Ley')
def get_desc(x):
    w = x['fex18'].sum()
    if w == 0: return pd.Series([0,0], index=['Escolaridad', 'Edad'])
    return pd.Series([(x['años_escolaridad']*x['fex18']).sum()/w, (x['edad']*x['fex18']).sum()/w], index=['Escolaridad', 'Edad'])
desc = df_f.groupby(['young', 'Periodo']).apply(get_desc).reset_index()
st.dataframe(desc.sort_values(['young', 'Periodo'], ascending=[True, False]), use_container_width=True, hide_index=True)
