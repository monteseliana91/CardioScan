"""
Aplicación Streamlit para predicción de supervivencia post-infarto (Ecocardiograma)
Comparación de Red Neuronal vs Regresión Logística
Incluye predicción individual y por lotes (CSV)
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import io
import warnings
warnings.filterwarnings('ignore')

# ── Configuración de Página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioScan · Echo",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #0b0f1a;
    --surface:   #111827;
    --surface2:  #1a2236;
    --border:    #1e2d45;
    --accent:    #ff4b4b;
    --accent2:   #7c3aed;
    --vivo:      #10b981;
    --muerto:    #ef4444;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --font-head: 'Syne', sans-serif;
    --font-body: 'DM Sans', sans-serif;
    --font-mono: 'DM Mono', monospace;
}

.stApp { background: var(--bg); color: var(--text); font-family: var(--font-body); }
.main .block-container { padding: 2rem 2.5rem; max-width: 1400px; }
#MainMenu, footer, header { visibility: hidden; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #1a0d0d 0%, #1a1040 50%, #0d0d0d 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3rem 3.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero-tag {
    font-family: var(--font-mono);
    font-size: 0.72rem; letter-spacing: 0.2em;
    color: var(--accent); text-transform: uppercase; margin-bottom: 0.8rem;
}
.hero h1 {
    font-family: var(--font-head); font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.6rem 0; line-height: 1.1;
}
.hero p { color: var(--muted); font-size: 1rem; max-width: 560px; margin: 0; line-height: 1.6; }

/* Metric cards */
.metric-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.2rem 1.5rem;
    position: relative; overflow: hidden;
}
.metric-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.metric-label { font-family: var(--font-mono); font-size: 0.68rem; letter-spacing: 0.15em; color: var(--muted); text-transform: uppercase; margin-bottom: 0.4rem; }
.metric-value { font-family: var(--font-head); font-size: 1.9rem; font-weight: 700; color: var(--accent); line-height: 1; }
.metric-sub { font-size: 0.78rem; color: var(--muted); margin-top: 0.3rem; }

/* Section title */
.section-title { font-family: var(--font-head); font-size: 1.1rem; font-weight: 700; color: var(--text); letter-spacing: 0.03em; margin: 0 0 1.2rem 0; display: flex; align-items: center; gap: 0.6rem; }
.section-title span.dot { width: 8px; height: 8px; background: var(--accent); border-radius: 50%; display: inline-block; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #cc3333) !important;
    color: #fff !important; font-family: var(--font-head) !important;
    font-weight: 700 !important; border: none !important;
    border-radius: 10px !important; padding: 0.6rem 1.6rem !important;
    letter-spacing: 0.04em !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 8px 20px rgba(255,75,75,0.25) !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }

/* Result badge */
.result-badge { display: inline-block; padding: 0.5rem 1.4rem; border-radius: 50px; font-family: var(--font-head); font-size: 1.4rem; font-weight: 800; letter-spacing: 0.05em; margin-top: 0.5rem; }
.badge-sobrevive { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid #10b981; }
.badge-fallecido { background: rgba(239,68,68,0.15);   color: #ef4444; border: 1px solid #ef4444; }

.batch-row {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 12px; padding: 0.9rem 1.4rem; margin-bottom: 0.6rem;
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;
}

hr { border-color: var(--border) !important; opacity: 0.3; }
[data-testid="stFileUploader"] { background: var(--surface2) !important; border: 1px dashed var(--border) !important; border-radius: 12px !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Constantes y Datos ────────────────────────────────────────────────────────
FEATURES = [
    "Age", "Pericardial Effusion", "Fractional Shortening", 
    "EPSS", "LVDD", "Wall Motion Index"
]
# Rangos calculados: (min, max, mean)
RANGOS = {
    "Age": (35.0, 86.0, 62.58),
    "Pericardial Effusion": (0.0, 1.0, 0.18),
    "Fractional Shortening": (0.01, 0.61, 0.22),
    "EPSS": (0.0, 40.0, 12.14),
    "LVDD": (2.32, 6.74, 4.76),
    "Wall Motion Index": (1.0, 3.0, 1.36)
}
ETIQUETAS = {0: "Fallecido", 1: "Sobrevive"}
COLOR_CLASE = {"Sobrevive": "#10b981", "Fallecido": "#ef4444"}

@st.cache_data
def cargar_datos():
    col_names = [
        "survival", "still_alive", "age_at_heart_attack", "pericardial_effusion",
        "fractional_shortening", "epss", "lvdd", "wall_motion_score",
        "wall_motion_index", "mult", "name", "group", "alive_at_1"
    ]
    df = pd.read_csv('echocardiogram.data', names=col_names, na_values='?', on_bad_lines='skip')
    # Selección de columnas de interés
    cols_to_use = [
        "age_at_heart_attack", "pericardial_effusion", "fractional_shortening", 
        "epss", "lvdd", "wall_motion_index", "still_alive"
    ]
    df = df[cols_to_use]
    # Imputación de valores faltantes
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=cols_to_use)
    df_imputed['still_alive'] = df_imputed['still_alive'].round().astype(int)
    return df_imputed

# ── Entrenamiento ──────────────────────────────────────────────────────────────
@st.cache_resource
def entrenar_modelos():
    df = cargar_datos()
    X = df.drop(columns=["still_alive"])
    y = df["still_alive"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    # Red Neuronal
    mlp = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, random_state=42)
    mlp.fit(X_tr, y_train)
    pred_mlp = mlp.predict(X_te)

    # Regresión Logística
    logreg = LogisticRegression(random_state=42)
    logreg.fit(X_tr, y_train)
    pred_lr = logreg.predict(X_te)

    def metricas(y_true, y_pred):
        return {
            'accuracy':  accuracy_score(y_true, y_pred),
            'f1':        f1_score(y_true, y_pred, zero_division=0),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall':    recall_score(y_true, y_pred, zero_division=0),
            'cm':        confusion_matrix(y_true, y_pred),
            'report':    classification_report(y_true, y_pred, 
                                               target_names=["Fallecido", "Sobrevive"],
                                               output_dict=True, zero_division=0)
        }

    return {
        'mlp':    {'modelo': mlp, 'scaler': scaler, **metricas(y_test, pred_mlp)},
        'logreg': {'modelo': logreg, 'scaler': scaler, **metricas(y_test, pred_lr)},
        'X_test': X_test, 'y_test': y_test
    }

# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False,
                xticklabels=["Fallecido", "Sobrevive"],
                yticklabels=["Fallecido", "Sobrevive"],
                ax=ax, annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(title, color='#e2e8f0', fontsize=11, fontweight='bold', pad=14)
    ax.set_xlabel('Predicción', color='#64748b', fontsize=9)
    ax.set_ylabel('Real', color='#64748b', fontsize=9)
    ax.tick_params(colors='#64748b', labelsize=8)
    plt.tight_layout()
    return fig

# ── Main UI ───────────────────────────────────────────────────────────────────
def main():
    df = cargar_datos()
    modelos_data = entrenar_modelos()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:1rem 0 0.5rem'>
            <div style='font-size:2.5rem'>❤️</div>
            <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;color:#e2e8f0;margin-top:0.3rem'>CardioScan</div>
            <div style='font-family:"DM Mono",monospace;font-size:0.68rem;color:#ff4b4b;letter-spacing:0.2em'>ECHO-CARDIOGRAM</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**📊 Estadísticas Globales**")
        st.write(f"Pacientes: **{len(df)}**")
        
        counts = df['still_alive'].value_counts()
        for val, name in ETIQUETAS.items():
            cnt = counts.get(val, 0)
            col = COLOR_CLASE[name]
            st.markdown(f"<span style='color:{col}'>●</span> {name}: **{cnt}**", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**Test Set Results**")
        st.write(f"MLP Acc: **{modelos_data['mlp']['accuracy']*100:.1f}%**")
        st.write(f"LR Acc: **{modelos_data['logreg']['accuracy']*100:.1f}%**")

    # Hero
    st.markdown(f"""
    <div class="hero">
        <div class="hero-tag">🔬 Análisis de Supervivencia Cardíaca</div>
        <h1>Predicción de Supervivencia<br>Post-Infarto</h1>
        <p>Sistema inteligente para determinar la probabilidad de supervivencia 
        basado en indicadores de ecocardiografía clínica.</p>
    </div>""", unsafe_allow_html=True)

    # KPI Cards
    st.markdown('<p class="section-title"><span class="dot"></span>Rendimiento del Sistema (Test)</p>', unsafe_allow_html=True)
    m_mlp = modelos_data['mlp']
    m_lr = modelos_data['logreg']
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">Acc. MLP</div><div class="metric-value">{m_mlp["accuracy"]*100:.1f}%</div><div class="metric-sub">Red Neuronal</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="metric-label">Acc. LR</div><div class="metric-value">{m_lr["accuracy"]*100:.1f}%</div><div class="metric-sub">Reg. Logística</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div class="metric-label">F1 MLP</div><div class="metric-value">{m_mlp["f1"]:.2f}</div><div class="metric-sub">F1-Score</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><div class="metric-label">Precision MLP</div><div class="metric-value">{m_mlp["precision"]:.2f}</div><div class="metric-sub">Precision</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Secciones
    tab_ind, tab_batch = st.tabs(["🧬 Predicción Individual", "📦 Predicción por Lotes"])

    # 🧬 PREDICCIÓN INDIVIDUAL
    with tab_ind:
        st.markdown('<p class="section-title"><span class="dot"></span>Configuración de Paciente</p>', unsafe_allow_html=True)
        
        col_model, col_empty = st.columns([1, 2])
        with col_model:
            modelo_ind = st.selectbox("Seleccionar Modelo", ["Red Neuronal (MLP)", "Regresión Logística"], key="sel_mod_ind")
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_in1, col_in2, col_in3 = st.columns(3)
        inputs = {}
        for i, feature in enumerate(FEATURES):
            min_v, max_v, def_v = RANGOS[feature]
            curr_col = [col_in1, col_in2, col_in3][i % 3]
            with curr_col:
                st.markdown(f"<div style='font-family:DM Mono; font-size:0.8rem; color:#ff4b4b'>{feature}</div>", unsafe_allow_html=True)
                inputs[feature] = st.number_input(f"In_{feature}", min_value=float(min_v), max_value=float(max_v), value=float(def_v), format="%.2f", label_visibility="collapsed", key=f"ind_{feature}")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔬 Analizar Paciente", use_container_width=True):
            input_data = np.array([[inputs[f] for f in FEATURES]])
            key = 'mlp' if "Red Neuronal" in modelo_ind else 'logreg'
            scaler = modelos_data[key]['scaler']
            model = modelos_data[key]['modelo']
            pred = model.predict(scaler.transform(input_data))[0]
            
            res_text = ETIQUETAS[int(pred)]
            res_cls = "sobrevive" if pred == 1 else "fallecido"
            icon = "🟢" if pred == 1 else "🔴"
            
            st.markdown(f'<div style="text-align:center; margin-top:1rem"><div class="result-badge badge-{res_cls}">{icon} {res_text}</div></div>', unsafe_allow_html=True)

    # 📦 PREDICCIÓN POR LOTES
    with tab_batch:
        st.markdown('<p class="section-title"><span class="dot"></span>Procesamiento de Lotes (CSV)</p>', unsafe_allow_html=True)
        
        col_batch_mod, col_batch_info = st.columns([1, 2])
        with col_batch_mod:
            modelo_batch = st.selectbox("Seleccionar Modelo", ["Red Neuronal (MLP)", "Regresión Logística"], key="sel_mod_batch")
        with col_batch_info:
            st.info("El CSV debe contener las 6 variables clínicas + columna 'still_alive' para generar la matriz de confusión.")
            
        archivo = st.file_uploader("Cargar archivo CSV", type=["csv"])
        
        if archivo is not None:
            try:
                batch_df = pd.read_csv(archivo)
                # Validar columnas
                expected_cols = ["age_at_heart_attack", "pericardial_effusion", "fractional_shortening", "epss", "lvdd", "wall_motion_index"]
                
                if not all(col in batch_df.columns for col in expected_cols):
                    st.error("Error: El CSV no contiene todas las columnas requeridas.")
                    st.write("Columnas esperadas:", expected_cols)
                else:
                    # Imputar si hay nulos en el lote
                    imputer = SimpleImputer(strategy='mean')
                    X_batch = pd.DataFrame(imputer.fit_transform(batch_df[expected_cols]), columns=expected_cols)
                    
                    key = 'mlp' if "Red Neuronal" in modelo_batch else 'logreg'
                    scaler = modelos_data[key]['scaler']
                    model = modelos_data[key]['modelo']
                    
                    preds = model.predict(scaler.transform(X_batch))
                    
                    st.markdown("---")
                    col_res1, col_res2 = st.columns([1, 1.5])
                    
                    with col_res1:
                        st.markdown("**Resultados del Lote**")
                        counts_pred = pd.Series(preds).value_counts()
                        for val, name in ETIQUETAS.items():
                            c = counts_pred.get(val, 0)
                            col = COLOR_CLASE[name]
                            st.markdown(f"<div class='batch-row'><span>{name}</span><b style='color:{col}'>{c}</b></div>", unsafe_allow_html=True)
                    
                    with col_res2:
                        if 'still_alive' in batch_df.columns:
                            y_real = batch_df['still_alive'].fillna(0).round().astype(int)
                            cm_batch = confusion_matrix(y_real, preds)
                            acc_batch = accuracy_score(y_real, preds)
                            
                            st.markdown(f"**Matriz de Confusión (Lote)** · Acc: {acc_batch*100:.1f}%")
                            fig_batch = plot_confusion_matrix(cm_batch, f"Lote: {modelo_batch}")
                            st.pyplot(fig_batch)
                        else:
                            st.warning("Columna 'still_alive' no encontrada. No se puede generar la matriz de confusión.")
                    
                    st.markdown("**Vista previa de predicciones**")
                    results_df = batch_df.copy()
                    results_df['Predicción'] = [ETIQUETAS[p] for p in preds]
                    st.dataframe(results_df, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error al procesar el archivo: {e}")

    st.markdown("---")
    st.markdown('<p style="text-align:center; color:#64748b; font-size:0.7rem">CardioScan Engine v1.1 · Echocardiogram Analysis System</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
