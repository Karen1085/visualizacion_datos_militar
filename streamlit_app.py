import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# 1. CONFIGURACIÓN VISUAL (CYBERPUNK / NEON)
st.set_page_config(page_title="Estadísticas Ley 1780", layout="wide", initial_sidebar_state="expanded")

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
    div[data-testid="stMetricDelta"] > div { font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)

# --- FUNCIONES AUXILIARES ---
def format_num(num):
    """Abrevia números grandes para hacerlos más legibles"""
    if num >= 1e6: return f"{num/1e6:.2f}M"
    if num >= 1e3: return f"{num/1e3:.2f}k"
    return f"{num:.0f}"

# 2. CARGA DE DATOS
@st.cache_data
def load_data():
    try:
        df = pd.read_parquet("datos_tesis.parquet") 
    except:
        st.warning("No se encontró 'datos_tesis.parquet'. Usando datos de prueba.")
        df = pd.DataFrame()
        st.stop()
    
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # Limpieza de nulos en variables clave de conteo
    for col in ['ocupados', 'desocupados', 'inactivos']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    df['posicion_ocupacional'] = df['posicion_ocupacional'].astype('object').fillna('No aplica')
    df['ingreal'] = pd.to_numeric(df['ingreal'], errors='coerce').fillna(0)
    
    # Formateo de etiquetas para tamaño del hogar
    df['tamaño_hogar'] = df['tamaño_hogar'].astype(str).replace({
        '1': 'Unipersonal', '1.0': 'Unipersonal',
        '2': '2-3 Personas', '2.0': '2-3 Personas',
        '3': '4 o más', '3.0': '4 o más'
    })
    
    # Normalizar dummies
    df['formal_num'] = np.where(df['formal_ss'].astype(str).str.strip().str.lower() == 'formal', 1.0, 0.0)
    df['part_num'] = np.where(df['participa'].astype(str).str.strip().str.lower() == 'participa', 1.0, 0.0)
    return df

df = load_data()

# 3. PANEL LATERAL (FILTROS)
st.sidebar.markdown("### Filtros")

# Filtros Demográficos Generales
clase_opt = sorted(df['clase'].dropna().unique())
estrato_opt = sorted(df['estrato'].dropna().unique())
hijos_opt = sorted(df['hijos_05'].dropna().unique())
tam_hogar_opt = ["Unipersonal", "2-3 Personas", "4 o más"] # Orden lógico manual
asiste_opt = sorted(df['asiste_institucioneducativa'].dropna().unique())
nivel_opt = sorted(df['nivel_educ'].dropna().unique())

st.sidebar.markdown("**Filtros Poblacionales**")
clase_sel = st.sidebar.multiselect("Zona (Urbano/Rural)", clase_opt, default=clase_opt)
estrato_sel = st.sidebar.multiselect("Estrato", estrato_opt, default=estrato_opt)
nivel_sel = st.sidebar.multiselect("Nivel Educativo", nivel_opt, default=nivel_opt)

with st.sidebar.expander("Más filtros del hogar"):
    hijos_sel = st.sidebar.multiselect("Hijos 0-5 años", hijos_opt, default=hijos_opt)
    tam_hogar_sel = st.sidebar.multiselect("Tamaño del Hogar", tam_hogar_opt, default=tam_hogar_opt)
    asiste_sel = st.sidebar.multiselect("Asiste Inst. Educativa", asiste_opt, default=asiste_opt)

# Filtro Exclusivo Ocupados
st.sidebar.markdown("---")
st.sidebar.markdown("**Filtros de Ocupación** (Afectan Salario y Formalidad)")
pos_opt = [p for p in df['posicion_ocupacional'].unique() if p != 'No aplica']
pos_sel = st.sidebar.multiselect("Posición Ocupacional", pos_opt, default=pos_opt)

# Filtros de Tiempo
st.sidebar.markdown("---")
st.sidebar.markdown("**Configuración Temporal**")
ventana_sel = st.sidebar.radio("Ventana de Tiempo (Ref: Mayo 2016)", 
                               ["1. 6m antes y 6m después", "2. 12m antes y 12m después", "3. 24m antes y 24m después", "4. Todo el periodo"])

suavizado = st.sidebar.slider("Meses de Suavizado (Media Móvil)", 1, 12, 3)
st.sidebar.caption("Valores mayores suavizan ruido pero esconden cambios bruscos de corto plazo.")

# 4. PROCESAMIENTO DE DATOS
fecha_ley = pd.to_datetime('2016-05-01')

# A. Población total
df_geo = df[
    (df['clase'].isin(clase_sel)) & 
    (df['estrato'].isin(estrato_sel)) &
    (df['nivel_educ'].isin(nivel_sel)) &
    (df['hijos_05'].isin(hijos_sel)) &
    (df['tamaño_hogar'].isin(tam_hogar_sel)) &
    (df['asiste_institucioneducativa'].isin(asiste_sel))
].copy()

# B. Población específica para FORMALIDAD y SALARIOS
df_f = df_geo[df_geo['posicion_ocupacional'].isin(pos_sel)].copy()

def aplicar_ventana(data, ventana):
    if "1." in ventana: m = 6
    elif "2." in ventana: m = 12
    elif "3." in ventana: m = 24
    else: return data
    return data[(data['fecha'] >= fecha_ley - pd.DateOffset(months=m)) & (data['fecha'] <= fecha_ley + pd.DateOffset(months=m))]

df_geo = aplicar_ventana(df_geo, ventana_sel)
df_f = aplicar_ventana(df_f, ventana_sel)

# 5. CÁLCULOS PONDERADOS EN EL TIEMPO
def get_stats_geo(x):
    w = x['fex_c_x'].sum()
    if w <= 0: return pd.Series([0.0], index=['Participacion'])
    p = (x['part_num'] * x['fex_c_x']).sum() / w
    return pd.Series([p], index=['Participacion'])

def get_stats_f(x):
    w = x['fex_c_x'].sum()
    if w <= 0: return pd.Series([0.0, 0.0], index=['Formalidad', 'Salario'])
    f = (x['formal_num'] * x['fex_c_x']).sum() / w
    df_s = x[x['ingreal'] > 0]
    s = (df_s['ingreal'] * df_s['fex_c_x']).sum() / df_s['fex_c_x'].sum() if not df_s.empty else 0.0
    return pd.Series([f, s], index=['Formalidad', 'Salario'])

if not df_geo.empty:
    ts_part = df_geo.groupby(['fecha', 'young']).apply(get_stats_geo).reset_index()
    
    if not df_f.empty:
        ts_form = df_f.groupby(['fecha', 'young']).apply(get_stats_f).reset_index()
    else:
        ts_form = ts_part[['fecha', 'young']].copy()
        ts_form['Formalidad'] = 0.0
        ts_form['Salario'] = 0.0

    # Unir resultados
    ts = pd.merge(ts_part, ts_form, on=['fecha', 'young'], how='left').fillna(0)
    
    # Suavizado y Diferencias
    for col in ['Formalidad', 'Participacion', 'Salario']:
        ts[f'{col}_S'] = ts.groupby('young')[col].transform(lambda x: x.rolling(suavizado, min_periods=1).mean())
        ts[f'{col}_Diff'] = ts.groupby('young')[col].transform(lambda x: x.diff(12))
else:
    st.warning("La selección de filtros no arrojó resultados.")
    st.stop()


# 6. RESUMEN EJECUTIVO DINÁMICO & DELTAS
df_geo['Periodo'] = np.where(df_geo['fecha'] < fecha_ley, 'Pre-Ley', 'Post-Ley')
df_f['Periodo'] = np.where(df_f['fecha'] < fecha_ley, 'Pre-Ley', 'Post-Ley')

def calc_avg_period(data, col_val):
    if data.empty: return 0, 0
    pre = data[data['Periodo'] == 'Pre-Ley']
    post = data[data['Periodo'] == 'Post-Ley']
    
    val_pre = (pre[col_val] * pre['fex_c_x']).sum() / pre['fex_c_x'].sum() if pre['fex_c_x'].sum() > 0 else 0
    val_post = (post[col_val] * post['fex_c_x']).sum() / post['fex_c_x'].sum() if post['fex_c_x'].sum() > 0 else 0
    return val_pre, val_post

p_pre, p_post = calc_avg_period(df_geo, 'part_num')
f_pre, f_post = calc_avg_period(df_f, 'formal_num')

# Salario solo mayores a 0
df_sal_pre = df_f[(df_f['Periodo'] == 'Pre-Ley') & (df_f['ingreal'] > 0)]
df_sal_post = df_f[(df_f['Periodo'] == 'Post-Ley') & (df_f['ingreal'] > 0)]
s_pre = (df_sal_pre['ingreal'] * df_sal_pre['fex_c_x']).sum() / df_sal_pre['fex_c_x'].sum() if df_sal_pre['fex_c_x'].sum() > 0 else 0
s_post = (df_sal_post['ingreal'] * df_sal_post['fex_c_x']).sum() / df_sal_post['fex_c_x'].sum() if df_sal_post['fex_c_x'].sum() > 0 else 0

delta_f = (f_post - f_pre) * 100
delta_p = (p_post - p_pre) * 100
delta_s = (s_post / s_pre) - 1 if s_pre > 0 else 0

dir_f = "aumentó" if delta_f > 0 else "se redujo"
dir_s = "incremento" if delta_s > 0 else "caída"
dir_p = "subió" if delta_p > 0 else "cayó"

hipotesis_text = "lo cual sugiere un posible efecto positivo de la política" if delta_f > 0 else "lo que invita a cuestionar la efectividad de la política en este subgrupo"

st.title("Estadísticas descriptivas y análisis Ley 1780 Art. 19 y 20")

st.markdown(f"""
""")
st.markdown("---")

# 7. KPIs PRINCIPALES
st.markdown("#### Indicadores Globales (Promedio Periodo)")
c1, c2, c3, c4, c5 = st.columns(5)

tot_exp = df_geo['fex_c_x'].sum()
tot_ocu = (df_geo['ocupados'] * df_geo['fex_c_x']).sum()
tot_des = (df_geo['desocupados'] * df_geo['fex_c_x']).sum()

c1.metric("Población Total", format_num(tot_exp))
c2.metric("Total Ocupados", format_num(tot_ocu))
c3.metric("Total Desocupados", format_num(tot_des))
c4.metric("Formalidad Promedio", f"{f_post*100:.1f}%", delta=f"{delta_f:.1f} p.p. vs pre-ley")
c5.metric("Ingreso Real", f"${format_num(s_post)}", delta=f"{delta_s*100:.1f}% vs pre-ley")

st.markdown("---")

# 8. GRÁFICOS
layout_ui = dict(
    paper_bgcolor='#170a29', plot_bgcolor='#170a29', 
    font=dict(color="#ffffff", size=11), # Todo el texto de los gráficos en blanco
    xaxis=dict(showgrid=False, color='#ffffff', title="Fecha"), 
    yaxis=dict(gridcolor='#2a1642', color='#ffffff', title="Valor"),
    legend=dict(font=dict(color="#ffffff"), orientation="h", y=-0.25, x=0.5, xanchor="center"),
    margin=dict(l=10, r=10, t=40, b=10)
)
colores = {'Hombres 18-24': '#00e5ff', 'Hombres 25-28': '#b400ff', 'Hombres 29-32': '#ff007f', 'Mujeres': '#39ff14'}

def dibujar_fila(metric, label, is_pct=True):
    st.markdown(f"### {label}")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        fig = px.line(ts, x='fecha', y=f'{metric}_S', color='young', 
                      title=f"Nivel de {label}", color_discrete_map=colores)
        fig.add_vline(x=fecha_ley.timestamp()*1000, line_dash="dash", line_color="#39ff14")
        fig.update_layout(**layout_ui)
        if is_pct: fig.update_yaxes(tickformat=".1%", range=[0, 1] if metric=="Participacion" else None)
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        fig = px.line(ts.dropna(subset=[f'{metric}_Diff']), x='fecha', y=f'{metric}_Diff', color='young', 
                      title=f"Cambio YoY de {label}", color_discrete_map=colores)
        fig.add_vline(x=fecha_ley.timestamp()*1000, line_dash="dash", line_color="#39ff14")
        fig.update_layout(**layout_ui)
        if is_pct: fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)
        
    with c3:
        pivot = ts.pivot(index='fecha', columns='young', values=f'{metric}_S')
        if 'Hombres 18-24' in pivot.columns and 'Hombres 25-28' in pivot.columns:
            # Cálculo exacto del DiD
            pivot['Gap'] = pivot['Hombres 18-24'] - pivot['Hombres 25-28']
            gap_pre = pivot.loc[pivot.index < fecha_ley, 'Gap'].mean()
            gap_post = pivot.loc[pivot.index >= fecha_ley, 'Gap'].mean()
            did_est = gap_post - gap_pre 
            
            did_text = f"{did_est*100:.2f} p.p." if is_pct else f"${did_est:,.0f}"
            
            fig = px.bar(pivot.reset_index(), x='fecha', y='Gap', 
                         title=f"Brecha (Tratamiento - Control)<br><sup>Efecto DiD Promedio: {did_text}</sup>", 
                         color='Gap', color_continuous_scale=['#ff007f', '#00e5ff'])
            
            # Línea de la Ley
            fig.add_vline(x=fecha_ley.timestamp()*1000, line_dash="dash", line_color="#39ff14")
            
            # Líneas de los promedios Pre y Post
            fig.add_shape(type="line", x0=pivot.index.min(), x1=fecha_ley, y0=gap_pre, y1=gap_pre, 
                          line=dict(color="white", width=2, dash="dot"))
            fig.add_shape(type="line", x0=fecha_ley, x1=pivot.index.max(), y0=gap_post, y1=gap_post, 
                          line=dict(color="yellow", width=2, dash="dot"))
            
            fig.update_layout(**layout_ui, coloraxis_showscale=False)
            if is_pct: fig.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig, use_container_width=True)
            
    st.markdown("<br>", unsafe_allow_html=True)

# Renderizar Filas
dibujar_fila("Participacion", "Participación en el Mercado Laboral")
dibujar_fila("Formalidad", "Formalidad Laboral")
dibujar_fila("Salario", "Ingreso Laboral Real", is_pct=False)
