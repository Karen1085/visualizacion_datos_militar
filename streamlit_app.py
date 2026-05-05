import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# 1. CONFIGURACIÓN DE INTERFAZ CORPORATIVA (CYBERPUNK / NEON)
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
    div[data-testid="metric-container"] { background-color: #170a29; border-top: 3px solid #b400ff; padding: 15px; border-radius: 8px; }
    div[data-testid="stMetricLabel"] { color: #ffffff !important; text-transform: uppercase; }
    div[data-testid="stMetricValue"] > div { color: #00e5ff !important; font-size: 1.8rem !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("Impacto de la Ley 1780 (ProJoven) en el Mercado Laboral")
st.markdown("Análisis Descriptivo y Validación de Supuestos DiD - Factor de Expansión GEIH")
st.markdown("---")

# 2. CARGA DE DATOS
@st.cache_data
def load_data():
    try:
        df = pd.read_parquet("datos_tesis.parquet")
    except:
        st.error("Archivo 'datos_tesis.parquet' no encontrado. Verifica que esté en la misma carpeta.")
        st.stop()
        
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # Limpieza extrema de texto para Participación y Formalidad
    df['formal_num'] = np.where(df['formal_ss'].astype(str).str.strip().str.lower() == 'formal', 1.0, 0.0)
    df['part_num'] = np.where(df['part_mercadol'].astype(str).str.strip().str.lower() == 'participa', 1.0, 0.0)
    return df

df = load_data()

# 3. PANEL LATERAL (FILTROS Y VENTANAS EXACTAS)
st.sidebar.markdown("### 🔍 Filtros Territoriales")
clase_options = sorted(df['clase'].dropna().astype(str).unique())
area_options = sorted(df['area'].dropna().astype(str).unique())
posicion_options = sorted(df['posicion_ocup'].dropna().astype(str).unique())

clase_sel = st.sidebar.multiselect("Zona (Clase)", options=clase_options, default=clase_options)
area_sel = st.sidebar.multiselect("Áreas Metropolitanas", options=area_options, default=area_options[:5])

st.sidebar.markdown("### 💼 Filtro de Ocupación")
default_pos = ["Asalariados (Empresa/Gobierno)"] if "Asalariados (Empresa/Gobierno)" in posicion_options else posicion_options[:1]
posicion_sel = st.sidebar.multiselect("Posición Ocupacional (Para Formalidad)", options=posicion_options, default=default_pos)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⏱️ Análisis de Sensibilidad (Bandwidth)")
ventana_sel = st.sidebar.radio("Periodo de Evaluación (Política: Mayo 2016)", [
    "1. 6m antes y 6m después", 
    "2. 12m antes y 12m después", 
    "3. 24m antes y 24m después",
    "4. Todo el periodo de análisis"
])
suavizado = st.sidebar.slider("Suavizado (Media Móvil)", 1, 12, 3)

# 4. LÓGICA DE FILTRADO SEPARADA (Para arreglar el gráfico de Participación)
# df_geo: Población total (ocupados + inactivos) para medir PARTICIPACIÓN
df_geo = df[(df['clase'].astype(str).isin(clase_sel)) & (df['area'].astype(str).isin(area_sel))].copy()

# df_f: Solo ocupados filtrados por posición para medir FORMALIDAD
df_f = df_geo[df_geo['posicion_ocup'].astype(str).isin(posicion_sel)].copy()

fecha_ley = pd.to_datetime('2016-05-01')

def aplicar_ventana(data, ventana):
    if "1." in ventana:
        return data[(data['fecha'] >= fecha_ley - pd.DateOffset(months=6)) & (data['fecha'] <= fecha_ley + pd.DateOffset(months=6))]
    elif "2." in ventana:
        return data[(data['fecha'] >= fecha_ley - pd.DateOffset(months=12)) & (data['fecha'] <= fecha_ley + pd.DateOffset(months=12))]
    elif "3." in ventana:
        return data[(data['fecha'] >= fecha_ley - pd.DateOffset(months=24)) & (data['fecha'] <= fecha_ley + pd.DateOffset(months=24))]
    return data # Todo el periodo

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

if not df_geo.empty:
    # 1. Calcular participación con df_geo (total)
    ts_part = df_geo.groupby(['fecha', 'young']).apply(calc_participacion, include_groups=False).reset_index()
    
    # 2. Calcular formalidad con df_f (filtrado)
    if not df_f.empty:
        ts_form = df_f.groupby(['fecha', 'young']).apply(calc_formal_salario, include_groups=False).reset_index()
    else:
        ts_form = pd.DataFrame(columns=['fecha', 'young', 'Tasa_Formalidad', 'Salario_Real'])
    
    # 3. Unir bases (outer join para que no colapse si un grupo no tiene ocupados un mes)
    ts_data = pd.merge(ts_part, ts_form, on=['fecha', 'young'], how='outer').fillna(0)
    
    # 4. Suavizados y Diferencias
    for target in ['Tasa_Formalidad', 'Tasa_Participacion', 'Salario_Real']:
        ts_data[f'{target}_S'] = ts_data.groupby('young')[target].transform(lambda x: x.rolling(suavizado, min_periods=1).mean())
        if target != 'Tasa_Participacion':
            ts_data[f'{target}_Diff'] = ts_data.groupby('young')[target].transform(lambda x: x.diff(12))
            
    # 5. Brecha DiD (Tratamiento - Control)
    pivot_form = ts_data.pivot(index='fecha', columns='young', values='Tasa_Formalidad_S').reset_index()
    if 'Hombres 18-24' in pivot_form.columns and 'Hombres 25-28' in pivot_form.columns:
        pivot_form['Brecha (Tratamiento - Control)'] = pivot_form['Hombres 18-24'] - pivot_form['Hombres 25-28']
else:
    st.warning("No hay datos para estos filtros.")
    st.stop()

# 6. CONFIGURACIÓN VISUAL GENERAL
layout_ui = dict(
    paper_bgcolor='#170a29', plot_bgcolor='#170a29', font=dict(color="#ffffff", size=12),
    xaxis=dict(showgrid=False, color='#ffffff'), yaxis=dict(showgrid=True, gridcolor='#2a1642', color='#ffffff'),
    legend=dict(font=dict(color="#ffffff"), orientation="h", y=-0.2, x=0.5, xanchor="center")
)
colores = {'Hombres 18-24': '#00e5ff', 'Hombres 25-28': '#b400ff', 'Hombres 29-32': '#ff007f', 'Mujeres': '#39ff14'}

# 7. KPIs SUPERIORES
k1, k2, k3, k4 = st.columns(4)
k1.metric("Pob. Expandida (Total)", f"{int(df_geo['fex18'].sum()):,}")
k2.metric("Muestra Filtrada (N)", f"{len(df_geo):,}")
k3.metric("Formalidad Promedio", f"{ts_data['Tasa_Formalidad'].mean()*100:.1f}%")
k4.metric("Participación (PEA) Promedio", f"{ts_data['Tasa_Participacion'].mean()*100:.1f}%")

# 8. GRÁFICAS PRINCIPALES
c1, c2 = st.columns(2)
with c1:
    st.markdown("#### 1. Tasa de Formalidad")
    fig1 = px.line(ts_data, x='fecha', y='Tasa_Formalidad_S', color='young', color_discrete_map=colores)
    fig1.add_vline(x=fecha_ley.timestamp()*1000, line_dash="dash", line_color="#39ff14")
    fig1.update_layout(**layout_ui); fig1.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.markdown("#### 2. Tasa de Participación (PEA)")
    fig2 = px.line(ts_data, x='fecha', y='Tasa_Participacion_S', color='young', color_discrete_map=colores)
    fig2.add_vline(x=fecha_ley.timestamp()*1000, line_dash="dash", line_color="#39ff14")
    fig2.update_layout(**layout_ui)
    # ESCALA FORZADA DE 0 A 100% PARA EVITAR QUE SE VEA PLANA
    fig2.update_yaxes(tickformat=".1%", range=[0, 1.0])
    st.plotly_chart(fig2, use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.markdown("#### 3. Cambios YoY Formalidad (Elimina Estacionalidad)")
    fig3 = px.line(ts_data.dropna(subset=['Tasa_Formalidad_Diff']), x='fecha', y='Tasa_Formalidad_Diff', color='young', color_discrete_map=colores)
    fig3.add_vline(x=fecha_ley.timestamp()*1000, line_dash="dash", line_color="#39ff14")
    fig3.update_layout(**layout_ui)
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    st.markdown("#### 4. Salario Real Promedio (Ocupados)")
    fig4 = px.line(ts_data, x='fecha', y='Salario_Real_S', color='young', color_discrete_map=colores)
    fig4.add_vline(x=fecha_ley.timestamp()*1000, line_dash="dash", line_color="#39ff14")
    fig4.update_layout(**layout_ui); fig4.update_yaxes(tickformat="$,.0f")
    st.plotly_chart(fig4, use_container_width=True)

# 9. SECCIÓN ECONOMÉTRICA (DiD)
st.markdown("---")
st.markdown("### 📊 Validación Econométrica: Diferencias en Diferencias (DiD)")

c5, c6 = st.columns([2, 1])

with c5:
    st.markdown("#### 5. Brecha de Formalidad (Tratamiento vs Control Principal)")
    if 'Brecha (Tratamiento - Control)' in pivot_form.columns:
        fig5 = px.bar(pivot_form, x='fecha', y='Brecha (Tratamiento - Control)', 
                      color='Brecha (Tratamiento - Control)', color_continuous_scale=['#ff007f', '#00e5ff'])
        fig5.add_vline(x=fecha_ley.timestamp()*1000, line_dash="dash", line_color="#39ff14", annotation_text="Ley 1780")
        fig5.update_layout(**layout_ui, coloraxis_showscale=False)
        fig5.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Asegúrate de incluir 'Hombres 18-24' y 'Hombres 25-28' para ver la brecha.")

with c6:
    st.markdown("#### Controles Estructurales (Pre/Post Ley)")
    # Calculamos la demografía general usando df_geo para no sesgar
    df_geo['Periodo'] = np.where(df_geo['fecha'] < fecha_ley, 'Pre-Ley', 'Post-Ley')
    def get_c(x):
        w = x['fex18'].sum()
        if w == 0: return pd.Series([0,0], index=['Escolaridad', 'Edad'])
        return pd.Series([(x['años_escolaridad']*x['fex18']).sum()/w, (x['edad']*x['fex18']).sum()/w], index=['Escolaridad', 'Edad'])
    desc = df_geo.groupby(['young', 'Periodo']).apply(get_c, include_groups=False).reset_index()
    desc['Escolaridad'] = desc['Escolaridad'].round(2)
    desc['Edad'] = desc['Edad'].round(1)
    st.dataframe(desc.sort_values(['young', 'Periodo'], ascending=[True, False]), use_container_width=True, hide_index=True)
