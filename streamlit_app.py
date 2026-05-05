import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# 1. CONFIGURACIÓN VISUAL (CYBERPUNK / NEON)
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
        padding: 15px; border-radius: 8px;
    }
    div[data-testid="stMetricLabel"] { color: #ffffff !important; text-transform: uppercase; font-size: 0.8rem !important; }
    div[data-testid="stMetricValue"] > div { color: #00e5ff !important; font-size: 1.7rem !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# 2. CARGA DE DATOS
@st.cache_data
def load_data():
    try:
        df = pd.read_parquet("datos_tesis.parquet")
    except:
        st.error("No se encontró el archivo 'datos_tesis.parquet'.")
        st.stop()
    
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['formal_num'] = np.where(df['formal_ss'].astype(str).str.strip().str.lower() == 'formal', 1.0, 0.0)
    df['part_num'] = np.where(df['part_mercadol'].astype(str).str.strip().str.lower() == 'participa', 1.0, 0.0)
    return df

df = load_data()

# 3. PANEL LATERAL (FILTROS DE LA TESIS)
st.sidebar.markdown("### 🔍 Filtros de la Tesis")

# Espaciales
clase_options = sorted(df['clase'].dropna().unique())
area_options = sorted(df['area'].dropna().unique())
dpto_options = sorted(df['dpto'].dropna().unique())

clase_sel = st.sidebar.multiselect("Zona (Urbano/Rural)", clase_options, default=clase_options)
area_sel = st.sidebar.multiselect("Áreas Metropolitanas", area_options, default=area_options[:5])
dpto_sel = st.sidebar.multiselect("Departamento", dpto_options)

# Socioeconómicos
rol_sel = st.sidebar.multiselect("Rol en el Hogar", options=df['rol_hogar'].unique(), default=df['rol_hogar'].unique())
est_sel = st.sidebar.multiselect("Estado Civil", options=df['estado_civil'].unique(), default=df['estado_civil'].unique())
pos_sel = st.sidebar.multiselect("Posición Ocupacional", options=df['posicion_ocup'].unique(), default=["Asalariados (Empresa/Gobierno)"])

st.sidebar.markdown("---")
ventana_sel = st.sidebar.radio("Periodo de Evaluación", [
    "1. 6m antes y 6m después", "2. 12m antes y 12m después", "3. 24m antes y 24m después", "4. Todo el periodo"
])
suavizado = st.sidebar.slider("Meses de Suavizado", 1, 12, 3)

# 4. LÓGICA DE FILTRADO
# df_geo: Para Participación (Población total)
df_geo = df[(df['clase'].isin(clase_sel)) & (df['area'].isin(area_sel)) & 
            (df['rol_hogar'].isin(rol_sel)) & (df['estado_civil'].isin(est_sel))].copy()

if dpto_sel: df_geo = df_geo[df_geo['dpto'].isin(dpto_sel)]

# df_f: Para Formalidad y Salarios (Filtrado por Ocupación)
df_f = df_geo[df_geo['posicion_ocup'].isin(pos_sel)].copy()

fecha_ley = pd.to_datetime('2016-05-01')
def aplicar_ventana(data, ventana):
    if "1." in ventana: m = 6
    elif "2." in ventana: m = 12
    elif "3." in ventana: m = 24
    else: return data
    return data[(data['fecha'] >= fecha_ley - pd.DateOffset(months=m)) & (data['fecha'] <= fecha_ley + pd.DateOffset(months=m))]

df_geo = aplicar_ventana(df_geo, ventana_sel)
df_f = aplicar_ventana(df_f, ventana_sel)

# 5. CÁLCULOS PONDERADOS
def calc_stats(x):
    w = x['fex18'].sum()
    if w == 0: return pd.Series([0.0]*3, index=['Formalidad', 'Participacion', 'Salario'])
    
    part = (x['part_num'] * x['fex18']).sum() / w
    form = (x['formal_num'] * x['fex18']).sum() / w
    df_s = x[x['inglabo_real'] > 0]
    sal = (df_s['inglabo_real'] * df_s['fex18']).sum() / df_s['fex18'].sum() if not df_s.empty else 0.0
    return pd.Series([form, part, sal], index=['Formalidad', 'Participacion', 'Salario'])

if not df_geo.empty:
    ts_part = df_geo.groupby(['fecha', 'young']).apply(calc_stats).reset_index()
    ts_form = df_f.groupby(['fecha', 'young']).apply(calc_stats).reset_index()
    
    # Unimos para tener formalidad/salario (de df_f) y participacion (de df_geo)
    ts = pd.merge(ts_form[['fecha', 'young', 'Formalidad', 'Salario']], 
                  ts_part[['fecha', 'young', 'Participacion']], on=['fecha', 'young'])
    
    for col in ['Formalidad', 'Participacion', 'Salario']:
        ts[f'{col}_S'] = ts.groupby('young')[col].transform(lambda x: x.rolling(suavizado, min_periods=1).mean())
        ts[f'{col}_Diff'] = ts.groupby('young')[col].transform(lambda x: x.diff(12))
else:
    st.stop()

# 6. DASHBOARD - 3 GRÁFICAS POR LÍNEA
st.title("Impacto Ley 1780: Evaluación de Resultados")

# KPIs SUPERIORES
k1, k2, k3, k4 = st.columns(4)
k1.metric("Población Expandida", f"{int(df_geo['fex18'].sum()):,}")

# Ingreso Laboral Promedio Real
df_sal_ocup = df_f[df_f['inglabo_real'] > 0]
if not df_sal_ocup.empty:
    sal_prom = (df_sal_ocup['inglabo_real'] * df_sal_ocup['fex18']).sum() / df_sal_ocup['fex18'].sum()
else:
    sal_prom = 0
k2.metric("Ingreso Laboral Promedio", f"${sal_prom:,.0f}")

k3.metric("Formalidad Promedio", f"{(ts['Formalidad'].mean()*100):.1f}%")
k4.metric("Participación (PEA)", f"{(ts['Participacion'].mean()*100):.1f}%")

layout_ui = dict(
    paper_bgcolor='#170a29', plot_bgcolor='#170a29', font=dict(color="#ffffff", size=11),
    xaxis=dict(showgrid=False, color='#ffffff'), yaxis=dict(gridcolor='#2a1642', color='#ffffff'),
    legend=dict(font=dict(color="#ffffff"), orientation="h", y=-0.25, x=0.5, xanchor="center"),
    margin=dict(l=10, r=10, t=40, b=10)
)
colores = {'Hombres 18-24': '#00e5ff', 'Hombres 25-28': '#b400ff', 'Hombres 29-32': '#ff007f', 'Mujeres': '#39ff14'}

def dibujar_fila(metric, label, is_pct=True):
    st.markdown(f"#### Análisis de {label}")
    c1, c2, c3 = st.columns(3)
    
    # 1. Niveles
    with c1:
        fig = px.line(ts, x='fecha', y=f'{metric}_S', color='young', color_discrete_map=colores, title=f"Nivel de {label}")
        fig.add_vline(x=fecha_ley.timestamp()*1000, line_dash="dash", line_color="#39ff14")
        fig.update_layout(**layout_ui)
        if is_pct: fig.update_yaxes(tickformat=".1%", range=[0, 1] if metric=="Participacion" else None)
        st.plotly_chart(fig, use_container_width=True)
    
    # 2. Cambios YoY (t vs t-12)
    with c2:
        fig = px.line(ts.dropna(subset=[f'{metric}_Diff']), x='fecha', y=f'{metric}_Diff', color='young', color_discrete_map=colores, title=f"Cambio YoY en {label}")
        fig.add_vline(x=fecha_ley.timestamp()*1000, line_dash="dash", line_color="#39ff14")
        fig.update_layout(**layout_ui)
        if is_pct: fig.update_yaxes(tickformat=".2f") # Puntos porcentuales
        st.plotly_chart(fig, use_container_width=True)
        
    # 3. Validación DiD (Brecha Tratamiento vs Control)
    with c3:
        pivot = ts.pivot(index='fecha', columns='young', values=f'{metric}_S')
        if 'Hombres 18-24' in pivot.columns and 'Hombres 25-28' in pivot.columns:
            pivot['Gap'] = pivot['Hombres 18-24'] - pivot['Hombres 25-28']
            fig = px.bar(pivot.reset_index(), x='fecha', y='Gap', title=f"Brecha DiD (Trat. vs Cont.)", color='Gap', color_continuous_scale=['#ff007f', '#00e5ff'])
            fig.add_vline(x=fecha_ley.timestamp()*1000, line_dash="dash", line_color="#39ff14")
            fig.update_layout(**layout_ui, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

# Dibujar las 3 filas objetivo
dibujar_fila("Formalidad", "Formalidad Laboral")
dibujar_fila("Salario", "Ingreso Laboral Real", is_pct=False)
dibujar_fila("Participacion", "Participación (PEA)")

st.markdown("---")
st.markdown("#### Estadísticas Descriptivas Ponderadas")
df_geo['Periodo'] = np.where(df_geo['fecha'] < fecha_ley, 'Pre-Ley', 'Post-Ley')

def get_desc(x):
    w = x['fex18'].sum()
    if w == 0: return pd.Series([0.0, 0.0], index=['Escolaridad', 'Edad'])
    return pd.Series([(x['años_escolaridad']*x['fex18']).sum()/w, (x['edad']*x['fex18']).sum()/w], index=['Escolaridad', 'Edad'])

desc_table = df_geo.groupby(['young', 'Periodo']).apply(get_desc).reset_index()
st.dataframe(desc_table.sort_values(['young', 'Periodo'], ascending=[True, False]), use_container_width=True, hide_index=True)
