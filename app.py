import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Configuração da página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesPredict | Grupo 4",
    page_icon="🩺",
    layout="centered"
)

# ── Estilo CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.main { background-color: #f7f9fc; }

.header-box {
    background: linear-gradient(135deg, #1a3a5c 0%, #2d6a9f 100%);
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    color: white;
}
.header-box h1 {
    font-size: 2rem;
    font-weight: 600;
    margin: 0 0 0.3rem 0;
    font-family: 'IBM Plex Mono', monospace;
}
.header-box p {
    margin: 0;
    opacity: 0.85;
    font-size: 0.95rem;
}

.section-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #2d6a9f;
    margin: 1.5rem 0 0.75rem 0;
    border-left: 3px solid #2d6a9f;
    padding-left: 0.6rem;
}

.result-positive {
    background: #fff3f3;
    border: 2px solid #e05252;
    border-radius: 10px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-positive h2 { color: #c0392b; margin: 0 0 0.4rem 0; font-size: 1.6rem; }
.result-positive p  { color: #6b2020; margin: 0; font-size: 0.95rem; }

.result-negative {
    background: #f3fff6;
    border: 2px solid #3aab5e;
    border-radius: 10px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-negative h2 { color: #1e7a3e; margin: 0 0 0.4rem 0; font-size: 1.6rem; }
.result-negative p  { color: #1a4d2b; margin: 0; font-size: 0.95rem; }

.prob-bar-label {
    font-size: 0.8rem;
    color: #555;
    margin-bottom: 0.2rem;
    font-family: 'IBM Plex Mono', monospace;
}
.disclaimer {
    background: #fffbe6;
    border-left: 4px solid #f0c040;
    border-radius: 6px;
    padding: 0.8rem 1.2rem;
    font-size: 0.82rem;
    color: #6b5900;
    margin-top: 2rem;
}
.model-badge {
    display: inline-block;
    background: #eaf2fb;
    color: #1a3a5c;
    border-radius: 6px;
    padding: 0.2rem 0.7rem;
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 0.4rem;
}
.info-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 0.5rem;
    flex-wrap: wrap;
}
</style>
""", unsafe_allow_html=True)

# ── Cabeçalho ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🩺 DiabetesPredict</h1>
    <p>Previsão de Diabetes com Machine Learning — Pima Indians Diabetes Database</p>
    <span class="model-badge">Random Forest Otimizado · class_weight=balanced · GridSearchCV</span>
</div>
""", unsafe_allow_html=True)

# ── Carregamento do modelo ───────────────────────────────────────────────────
@st.cache_resource
def carregar_modelo():
    caminho = os.path.join(os.path.dirname(__file__), 'model', 'modelo_final.joblib')
    return joblib.load(caminho)

try:
    modelo = carregar_modelo()
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")
    st.stop()

# ── Formulário de entrada ────────────────────────────────────────────────────
st.markdown('<div class="section-title">Dados da Paciente</div>', unsafe_allow_html=True)
st.caption("Preencha os campos clínicos abaixo. Todos os valores devem ser numéricos.")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Número de Gestações", min_value=0, max_value=20, value=1, step=1,
        help="Quantas vezes a paciente esteve grávida"
    )
    glucose = st.number_input(
        "Glicose (mg/dL)", min_value=0, max_value=300, value=120, step=1,
        help="Concentração de glicose no plasma (teste oral de tolerância à glicose)"
    )
    blood_pressure = st.number_input(
        "Pressão Arterial Diastólica (mmHg)", min_value=0, max_value=150, value=70, step=1,
        help="Pressão arterial diastólica"
    )
    skin_thickness = st.number_input(
        "Espessura da Pele (mm)", min_value=0, max_value=100, value=20, step=1,
        help="Espessura da dobra cutânea do tríceps"
    )

with col2:
    insulin = st.number_input(
        "Insulina — 2h (mu U/ml)", min_value=0, max_value=900, value=80, step=1,
        help="Insulina sérica 2 horas após o teste"
    )
    bmi = st.number_input(
        "IMC (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, step=0.1,
        format="%.1f",
        help="Índice de Massa Corporal = peso / altura²"
    )
    dpf = st.number_input(
        "Função Pedigree do Diabetes", min_value=0.0, max_value=3.0, value=0.47, step=0.01,
        format="%.3f",
        help="Função que modela a história familiar de diabetes"
    )
    age = st.number_input(
        "Idade (anos)", min_value=1, max_value=120, value=33, step=1,
        help="Idade da paciente em anos"
    )

# ── Botão de predição ────────────────────────────────────────────────────────
st.markdown("---")
predict_btn = st.button("🔍 Realizar Predição", use_container_width=True, type="primary")

if predict_btn:
    # Montar DataFrame com as mesmas colunas usadas no treino
    dados_entrada = pd.DataFrame([{
        'Pregnancies':             pregnancies,
        'Glucose':                 glucose,
        'BloodPressure':           blood_pressure,
        'SkinThickness':           skin_thickness,
        'Insulin':                 insulin,
        'BMI':                     bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age':                     age
    }])

    # Substituir zeros por NaN nas colunas suspeitas (igual ao pré-processamento do notebook)
    colunas_suspeitas = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    dados_entrada[colunas_suspeitas] = dados_entrada[colunas_suspeitas].replace(0, np.nan)

    # Predição
    predicao = modelo.predict(dados_entrada)[0]
    probabilidades = modelo.predict_proba(dados_entrada)[0]
    prob_diabetes = probabilidades[1]
    prob_normal   = probabilidades[0]

    # ── Resultado ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Resultado da Predição</div>', unsafe_allow_html=True)

    if predicao == 1:
        st.markdown(f"""
        <div class="result-positive">
            <h2>⚠️ Diagnóstico: Diabética</h2>
            <p>O modelo classificou esta paciente como <strong>positiva para diabetes</strong>
            com probabilidade de <strong>{prob_diabetes:.1%}</strong>.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-negative">
            <h2>✅ Diagnóstico: Não Diabética</h2>
            <p>O modelo classificou esta paciente como <strong>negativa para diabetes</strong>
            com probabilidade de <strong>{prob_normal:.1%}</strong>.</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Probabilidades ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Probabilidades por Classe</div>', unsafe_allow_html=True)

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown('<div class="prob-bar-label">Não Diabética</div>', unsafe_allow_html=True)
        st.progress(float(prob_normal))
        st.caption(f"{prob_normal:.1%}")
    with col_p2:
        st.markdown('<div class="prob-bar-label">Diabética</div>', unsafe_allow_html=True)
        st.progress(float(prob_diabetes))
        st.caption(f"{prob_diabetes:.1%}")

    # ── Dados inseridos ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Dados Inseridos</div>', unsafe_allow_html=True)
    dados_exibir = pd.DataFrame([{
        'Gestações': pregnancies,
        'Glicose': glucose,
        'Pressão Arterial': blood_pressure,
        'Espessura da Pele': skin_thickness,
        'Insulina': insulin,
        'IMC': bmi,
        'Pedigree Diabetes': dpf,
        'Idade': age
    }])
    st.dataframe(dados_exibir, use_container_width=True, hide_index=True)

    # ── Interpretação ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Interpretação</div>', unsafe_allow_html=True)

    if predicao == 1:
        st.info(
            "**O modelo indicou risco de diabetes.** "
            "As variáveis com maior peso neste modelo são **Glicose**, **IMC** e **Idade**. "
            "Este resultado deve ser interpretado como uma triagem de apoio e não substitui "
            "avaliação médica profissional."
        )
    else:
        st.info(
            "**O modelo não identificou risco de diabetes.** "
            "Ainda assim, valores elevados de glicose e IMC são fatores de risco conhecidos. "
            "Consulte um profissional de saúde para acompanhamento preventivo."
        )

# ── Aviso legal ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Aviso:</strong> Esta aplicação é um projeto acadêmico desenvolvido para fins educacionais.
    O resultado <strong>não constitui diagnóstico médico</strong>. Consulte sempre um profissional de saúde.
    Modelo treinado no <em>Pima Indians Diabetes Database</em> — população específica, resultados podem
    não generalizar para outros grupos.
</div>
""", unsafe_allow_html=True)

# ── Rodapé ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Grupo 4 · Alan Uesugui Uemura · Julio Cesar Plaza · Pedro Renat · Algoritmos de IA — UNIMAR 2026")
