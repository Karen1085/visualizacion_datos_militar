import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np 
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
    
    /* Estilos ajustados para la Tabla DiD (más compacta) */
    .did-table { width: 100%; border-collapse: collapse; text-align: center; color: white; font-size: 0.85rem; margin-top: 5px; }
    .did-table th { background-color: #2a1642; padding: 8px; border-bottom: 3px solid #b400ff; font-weight: bold; }
    .did-table td { padding: 8px; border-bottom: 1px solid #3d2063; }
    .did-table tr:hover { background-color: #1a0d2b; }
    .did-table .highlight { color: #00e5ff; font-weight: bold; }
    .did-table .base { color: #888888; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# --- FUNCIONES AUXILIARES ---
def format_num(num):
    if num >= 1e6: return f"{num/1e6:.2f}M"
    if num >= 1e3: return f"{num/1e3:.2f}k"
    return f"{num:.0f}"

# 2. CARGA DE DATOS
@st.cache_data(ttl=3600)
def load_data():
    try:
        url = "https://github.com/Karen1085/visualizacion_datos_militar/blob/main/datos_tesis.parquet"
        df = pd.read_parquet(url) 
    except:
        st.warning("No se encontró 'datos_tesis.parquet'. Usando datos de prueba.")
        df = pd.DataFrame()
        st.stop()
    
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    for col in ['ocupados', 'desocupados', 'inactivos']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    df['posicion_ocupacional'] = df['posicion_ocupacional'].astype('object').fillna('No aplica')
    
    df['tamaño_hogar'] = df['tamaño_hogar'].astype(str).replace({
        '1': 'Unipersonal', '1.0': 'Unipersonal',
        '2': '2-3 Personas', '2.0': '2-3 Personas',
        '3': '4 o más', '3.0': '4 o más'
    })
    
    # Manejo de la nueva variable de dominio (Áreas Metropolitanas)
    if 'dominio_dane' in df.columns:
        df['dominio_dane'] = df['dominio_dane'].astype(str).replace({
            '1': '13 Ciudades y AM', '1.0': '13 Ciudades y AM',
            '2': 'Otras 10 Ciudades', '2.0': 'Otras 10 Ciudades'
        })
        # Tratar nulos o valores inesperados como "Otro"
        condicion = df['dominio_dane'].isin(['13 Ciudades y AM', 'Otras 10 Ciudades'])
        df['dominio_dane'] = np.where(condicion, df['dominio_dane'], 'Otro')
    else:
        df['dominio_dane'] = 'Otro'
    
    # --- MANEJO DE NULOS ESTRICTO TIPO STATA ---
    df['ingreal'] = pd.to_numeric(df['ingreal'], errors='coerce')
    
    # Solo 1.0 para formal, 0.0 para informal. TODO LO DEMÁS ES NaN.
    condicion_formal = df['formal_ss'].astype(str).str.strip().str.lower()
    df['formal_num'] = np.where(condicion_formal == 'formal', 1.0, 
                                np.where(condicion_formal == 'informal', 0.0, np.nan))
                                
    df['part_num'] = np.where(df['participa'].astype(str).str.strip().str.lower() == 'participa', 1.0, 0.0)
    
    return df

df = load_data()

# 3. PANEL LATERAL (FILTROS)
st.sidebar.markdown("### Filtros")

clase_opt = sorted(df['clase'].dropna().unique())
estrato_opt = sorted(df['estrato'].dropna().unique())
dominio_opt = sorted(df['dominio_dane'].dropna().unique())
tam_hogar_opt = ["Unipersonal", "2-3 Personas", "4 o más"] 
nivel_opt = sorted(df['nivel_educ'].dropna().unique())

st.sidebar.markdown("**Filtros Poblacionales**")
clase_sel = st.sidebar.multiselect("Zona (Urbano/Rural)", clase_opt, default=clase_opt)
dominio_sel = st.sidebar.multiselect("Área Geográfica", dominio_opt, default=dominio_opt)
estrato_sel = st.sidebar.multiselect("Estrato", estrato_opt, default=estrato_opt)
nivel_sel = st.sidebar.multiselect("Nivel Educativo", nivel_opt, default=nivel_opt)

with st.sidebar.expander("Más filtros del hogar"):
    tam_hogar_sel = st.sidebar.multiselect("Tamaño del Hogar", tam_hogar_opt, default=tam_hogar_opt)

st.sidebar.markdown("---")
st.sidebar.markdown("**Filtros de Ocupación** (Afectan Salario y Formalidad)")
pos_opt = [p for p in df['posicion_ocupacional'].unique() if p != 'No aplica']
pos_sel = st.sidebar.multiselect("Posición Ocupacional", pos_opt, default=pos_opt)

st.sidebar.markdown("---")
st.sidebar.markdown("**Configuración Temporal**")
ventana_sel = st.sidebar.radio("Ventana de Tiempo (Ref: Mayo 2016)", 
                               ["1. 6m antes y 6m después", "2. 12m antes y 12m después", "3. 24m antes y 24m después", "4. Todo el periodo"])

suavizado = st.sidebar.slider("Meses de Suavizado (Media Móvil)", 1, 12, 3)

# 4. PROCESAMIENTO DE DATOS MACRO
fecha_ley = pd.to_datetime('2016-05-01')

df_geo = df[
    (df['clase'].isin(clase_sel)) & 
    (df['dominio_dane'].isin(dominio_sel)) &
    (df['estrato'].isin(estrato_sel)) &
    (df['nivel_educ'].isin(nivel_sel)) &
    (df['tamaño_hogar'].isin(tam_hogar_sel))
].copy()

df_f = df_geo[df_geo['posicion_ocupacional'].isin(pos_sel)].copy()

def aplicar_ventana(data, ventana):
    if "1." in ventana: m = 6
    elif "2." in ventana: m = 12
    elif "3." in ventana: m = 24
    else: return data
    return data[(data['fecha'] >= fecha_ley - pd.DateOffset(months=m)) & (data['fecha'] <= fecha_ley + pd.DateOffset(months=m))]

df_geo = aplicar_ventana(df_geo, ventana_sel)
df_f = aplicar_ventana(df_f, ventana_sel)

df_geo['Periodo'] = np.where(df_geo['fecha'] < fecha_ley, 'Pre-Ley', 'Post-Ley')
df_f['Periodo'] = np.where(df_f['fecha'] < fecha_ley, 'Pre-Ley', 'Post-Ley')

# 5. CÁLCULOS PARA SERIES DE TIEMPO
def get_stats_geo(x):
    w_p = x['fex'].sum()
    p = (x['part_num'] * x['fex']).sum() / w_p if w_p > 0 else np.nan
    return pd.Series([p], index=['Participacion'])

def get_stats_f(x):
    df_form = x.dropna(subset=['formal_num'])
    w_f = df_form['fex'].sum()
    f = (df_form['formal_num'] * df_form['fex']).sum() / w_f if w_f > 0 else np.nan
    
    df_sal = x.dropna(subset=['ingreal'])
    w_s = df_sal['fex'].sum()
    s = (df_sal['ingreal'] * df_sal['fex']).sum() / w_s if w_s > 0 else np.nan
    return pd.Series([f, s], index=['Formalidad', 'Salario'])

if not df_geo.empty:
    ts_part = df_geo.groupby(['fecha', 'young']).apply(get_stats_geo).reset_index()
    if not df_f.empty:
        ts_form = df_f.groupby(['fecha', 'young']).apply(get_stats_f).reset_index()
    else:
        ts_form = ts_part[['fecha', 'young']].copy()
        ts_form['Formalidad'] = np.nan
        ts_form['Salario'] = np.nan

    ts = pd.merge(ts_part, ts_form, on=['fecha', 'young'], how='left')
    
    for col in ['Formalidad', 'Participacion', 'Salario']:
        ts[f'{col}_S'] = ts.groupby('young')[col].transform(lambda x: x.rolling(suavizado, min_periods=1).mean())
else:
    st.warning("La selección de filtros no arrojó resultados.")
    st.stop()


# 6. HEADER
st.title("Estadísticas descriptivas y análisis Ley 1780 Art. 19 y 20")
st.markdown("---")

# 7. FUNCIÓN GENERADORA DE TABLAS DiD
def calc_did_table(metric, is_pct=True):
    if metric == "Participacion":
        data = df_geo.dropna(subset=['part_num']).copy()
        val_col = 'part_num'
    elif metric == "Formalidad":
        data = df_f.dropna(subset=['formal_num']).copy()
        val_col = 'formal_num'
    else:
        data = df_f.dropna(subset=['ingreal']).copy()
        val_col = 'ingreal'

    def w_mean(grp):
        return (grp[val_col] * grp['fex']).sum() / grp['fex'].sum() if grp['fex'].sum() > 0 else np.nan

    if data.empty: return {}
    agg = data.groupby(['young', 'Periodo']).apply(w_mean).unstack()
    
    for p in ['Pre-Ley', 'Post-Ley']:
        if p not in agg.columns: agg[p] = np.nan
        
    agg['Var'] = agg['Post-Ley'] - agg['Pre-Ley']
    
    def get_val(grupo, col): return agg.loc[grupo, col] if grupo in agg.index else np.nan
    
    ctrl_var = get_val('Hombres 29-32', 'Var')
    
    filas = [
        {"nombre": "Control (29-32 años)", "grupo": "Hombres 29-32"},
        {"nombre": "Tratamiento 2 (25-28)", "grupo": "Hombres 25-28"},
        {"nombre": "Tratamiento 1 (18-24)", "grupo": "Hombres 18-24"}
    ]
    
    html = f"""<table class='did-table'>
        <tr>
            <th>Cohorte de Edad</th>
            <th>Pre-Ley {'(%)' if is_pct else '($)'}</th>
            <th>Post-Ley {'(%)' if is_pct else '($)'}</th>
            <th>Variación {'(pp)' if is_pct else '($)'}</th>
            <th>DiD</th>
        </tr>
    """
    
    for f in filas:
        pre = get_val(f['grupo'], 'Pre-Ley')
        post = get_val(f['grupo'], 'Post-Ley')
        var = get_val(f['grupo'], 'Var')
        
        if is_pct:
            pre_str = f"{pre*100:.1f}" if pd.notnull(pre) else "--"
            post_str = f"{post*100:.1f}" if pd.notnull(post) else "--"
            var_str = f"{var*100:+.2f}" if pd.notnull(var) else "--"
            if f['grupo'] == 'Hombres 29-32':
                did_str = "<span class='base'>Línea Base</span>"
            else:
                did = (var - ctrl_var) * 100 if pd.notnull(var) and pd.notnull(ctrl_var) else np.nan
                did_str = f"<span class='highlight'>{did:+.2f}</span>" if pd.notnull(did) else "--"
        else:
            pre_str = f"${pre:,.0f}" if pd.notnull(pre) else "--"
            post_str = f"${post:,.0f}" if pd.notnull(post) else "--"
            var_str = f"{var:+.0f}" if pd.notnull(var) else "--"
            if f['grupo'] == 'Hombres 29-32':
                did_str = "<span class='base'>Línea Base</span>"
            else:
                did = (var - ctrl_var) if pd.notnull(var) and pd.notnull(ctrl_var) else np.nan
                did_str = f"<span class='highlight'>{did:+.0f}</span>" if pd.notnull(did) else "--"

        html += f"<tr><td>{f['nombre']}</td><td>{pre_str}</td><td>{post_str}</td><td>{var_str}</td><td>{did_str}</td></tr>"
        
    html += "</table>"
    return html

# 8. GRÁFICOS Y RENDERIZADO
layout_ui = dict(
    paper_bgcolor='#170a29', plot_bgcolor='#170a29', 
    font=dict(color="#ffffff", size=11),
    legend=dict(font=dict(color="#ffffff"), orientation="h", y=-0.2, x=0.5, xanchor="center"),
    margin=dict(l=10, r=10, t=30, b=10)
)

def dibujar_fila(metric, label, is_pct=True):
    st.markdown(f"### {label}")
    
    ts_1824 = ts[ts['young'] == 'Hombres 18-24'].dropna(subset=[f'{metric}_S'])
    ts_2528 = ts[ts['young'] == 'Hombres 25-28'].dropna(subset=[f'{metric}_S'])
    ts_2932 = ts[ts['young'] == 'Hombres 29-32'].dropna(subset=[f'{metric}_S'])
    
    # Escala global unificada
    all_vals = pd.concat([ts_1824[f'{metric}_S'], ts_2528[f'{metric}_S'], ts_2932[f'{metric}_S']])
    if not all_vals.empty:
        y_min, y_max = all_vals.min(), all_vals.max()
        margen = (y_max - y_min) * 0.1
        if margen == 0: margen = y_max * 0.1 if y_max != 0 else 0.1
        rango = [y_min - margen, y_max + margen]
    else:
        rango = [0, 1] if is_pct else None
        
    formato = ".1%" if is_pct else None

    c1, c2, c3 = st.columns([3.5, 3.5, 3]) 
    
    with c1:
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig1.add_trace(go.Scatter(x=ts_1824['fecha'], y=ts_1824[f'{metric}_S'], 
                                 name='Trat. 18-24', line=dict(color='#00e5ff', width=2)), secondary_y=False)
        fig1.add_trace(go.Scatter(x=ts_2932['fecha'], y=ts_2932[f'{metric}_S'], 
                                 name='Control 29-32', line=dict(color='#ff007f', width=2)), secondary_y=True)
        
        fig1.add_vline(x=fecha_ley.timestamp()*1000, line_dash="dash", line_color="#39ff14")
        
        # Actualizando los ejes Y con Títulos
        fig1.update_yaxes(title_text="Tratamiento 18-24", range=rango, tickformat=formato, secondary_y=False, showgrid=False)
        fig1.update_yaxes(title_text="Control 29-32", range=rango, tickformat=formato, secondary_y=True, showgrid=False)
        fig1.update_xaxes(showgrid=False)
        fig1.update_layout(**layout_ui, title="18-24 años vs Control")
        st.plotly_chart(fig1, use_container_width=True)
        
    with c2:
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x=ts_2528['fecha'], y=ts_2528[f'{metric}_S'], 
                                 name='Trat. 25-28', line=dict(color='#b400ff', width=2)), secondary_y=False)
        fig2.add_trace(go.Scatter(x=ts_2932['fecha'], y=ts_2932[f'{metric}_S'], 
                                 name='Control 29-32', line=dict(color='#ff007f', width=2)), secondary_y=True)
        
        fig2.add_vline(x=fecha_ley.timestamp()*1000, line_dash="dash", line_color="#39ff14")
        
        # Actualizando los ejes Y con Títulos
        fig2.update_yaxes(title_text="Tratamiento 25-28", range=rango, tickformat=formato, secondary_y=False, showgrid=False)
        fig2.update_yaxes(title_text="Control 29-32", range=rango, tickformat=formato, secondary_y=True, showgrid=False)
        fig2.update_xaxes(showgrid=False)
        fig2.update_layout(**layout_ui, title="25-28 años vs Control")
        st.plotly_chart(fig2, use_container_width=True)
        
    with c3:
        st.markdown("<br>", unsafe_allow_html=True) 
        tabla_html = calc_did_table(metric, is_pct)
        st.markdown(tabla_html, unsafe_allow_html=True)
        
    st.markdown("---")

# Renderizar las 3 métricas
dibujar_fila("Participacion", "Participación en el Mercado Laboral")
dibujar_fila("Formalidad", "Formalidad Laboral")
dibujar_fila("Salario", "Ingreso Laboral Real", is_pct=False)
