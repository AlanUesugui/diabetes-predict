# DiabetesPredict — Diagnóstico de Diabetes com Machine Learning

Projeto da disciplina de Algoritmos de Inteligência Artificial — UNIMAR 2026

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://diabetes-predict-e9r77dqeijdqkadjjlzab4.streamlit.app/)

---

## Integrantes

| Nome | RA |
|---|---|
| Alan Uesugui Uemura | 2084808 |
| Julio Cesar Plaza | 2046253 |
Grupo 4

---

## O que é esse projeto

Diabetes é uma doença que, quando não diagnosticada a tempo, pode causar complicações sérias. A ideia desse projeto foi treinar um modelo de Machine Learning capaz de prever se uma paciente tem diabetes com base em exames clínicos simples — sem precisar de exames invasivos.

Usamos o Pima Indians Diabetes Database, que tem dados de 768 mulheres da população Pima Indian, com variáveis como glicose, IMC, pressão arterial e insulina.

---

## Objetivo

Construir um classificador binário que prevê se uma paciente é diabética ou não, priorizando o Recall — ou seja, queremos errar menos para o lado de "não detectar quem tem diabetes", porque esse erro tem consequências clínicas mais graves.

---

## Dataset

**Pima Indians Diabetes Database**

- Fonte: National Institute of Diabetes and Digestive and Kidney Diseases
- Disponível em: [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- 768 amostras, 8 variáveis preditoras numéricas
- Variável-alvo: `Outcome` (0 = não diabética, 1 = diabética)
- Dataset desbalanceado: 65,1% não diabéticas / 34,9% diabéticas

| Variável | Descrição |
|---|---|
| `Pregnancies` | Número de gestações |
| `Glucose` | Glicose no plasma (mg/dL) |
| `BloodPressure` | Pressão arterial diastólica (mmHg) |
| `SkinThickness` | Espessura da pele — tríceps (mm) |
| `Insulin` | Insulina sérica 2h após o teste (mu U/ml) |
| `BMI` | Índice de Massa Corporal (kg/m²) |
| `DiabetesPedigreeFunction` | Histórico familiar de diabetes |
| `Age` | Idade (anos) |
| `Outcome` | Diagnóstico — variável-alvo |

---

## Tipo de problema

Classificação binária supervisionada.

---

## O que fizemos

1. Análise exploratória dos dados (distribuição das classes, zeros suspeitos, outliers, correlações)
2. Tratamento dos zeros biologicamente impossíveis — substituídos por NaN e imputados pela mediana
3. Pipeline sklearn para evitar data leakage
4. `class_weight='balanced'` para lidar com o desbalanceamento das classes
5. Validação cruzada estratificada com 5 folds
6. GridSearchCV para otimizar os hiperparâmetros do Random Forest
7. Teste de Friedman para verificar se as diferenças entre os modelos são estatisticamente significativas
8. Avaliação final no conjunto de teste isolado

---

## Modelos treinados

| Modelo | Detalhe |
|---|---|
| Regressão Logística | `class_weight='balanced'` |
| Gaussian Naive Bayes | sem suporte a class_weight |
| Random Forest | `class_weight='balanced'` |
| **Random Forest Otimizado** | melhores hiperparâmetros via GridSearchCV |

---

## Modelo escolhido

**Random Forest Otimizado**

Foi o que teve melhor F1-Score (0,6992) e AUC-ROC (0,8252) no conjunto de teste, e o menor número de Falsos Negativos — só 11 casos de diabetes não detectados, contra 16 da Regressão Logística e 17 do Random Forest sem otimização. Em contexto clínico, esse é o critério mais importante.

---

## Métricas de avaliação

Usamos Acurácia, Precisão, Recall, F1-Score e AUC-ROC. A métrica prioritária foi o Recall, por conta do contexto clínico.

---

## Resultados principais

| Modelo | Acurácia | Recall | F1 | AUC |
|---|---|---|---|---|
| Logistic Regression | 0,7338 | 0,7037 | 0,6496 | 0,8126 |
| Gaussian NB | 0,7013 | 0,6296 | 0,5965 | 0,7646 |
| Random Forest | 0,7532 | 0,6852 | 0,6607 | 0,8231 |
| **RF Otimizado** | **0,7597** | **0,7963** | **0,6992** | **0,8252** |

---

## Estrutura dos arquivos

```
diabetes-predict/
│
├── app.py
├── requirements.txt
├── README.md
│
├── notebooks/
│   └── notebook_atualizado.ipynb
│
├── model/
│   └── modelo_final.joblib
│   └── colunas.joblib
│
├── reports/
│   └── relatorio_atualizado.pdf
│
└── data/
    └── diabetes.csv
```

---

## Tecnologias

Python, pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, joblib, Streamlit.

---

## Como rodar o notebook

```bash
git clone https://github.com/AlanUesugui/diabetes-predict.git
cd diabetes-predict
pip install -r requirements.txt
jupyter notebook notebooks/notebook_atualizado.ipynb
```

Execute todas as células em ordem. O modelo será salvo automaticamente em `model/modelo_final.joblib`.

---

## Como rodar o app localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

O app abre no navegador em `http://localhost:8501`.

---

## App publicado

**[Acesse aqui](https://SEU_LINK_DO_STREAMLIT_AQUI)**

---

## Limitações

- Dataset pequeno (768 amostras) — o modelo pode não generalizar bem
- Dados de uma população específica (Pima Indian) — pode não funcionar igual para outros grupos
- Insulin e SkinThickness têm muitos zeros (48,7% e 29,6%) — mesmo após imputação, há incerteza
- Não substitui avaliação médica — é uma ferramenta de triagem

---

## Conclusão

O projeto mostrou que dá para construir um classificador útil para triagem de diabetes com dados clínicos simples. O Random Forest Otimizado foi o que melhor equilibrou detecção de casos positivos e precisão, especialmente depois de tratar o desbalanceamento com `class_weight='balanced'` e otimizar os hiperparâmetros com GridSearchCV.

---

*UNIMAR 2026 — Algoritmos de Inteligência Artificial — Grupo 4*
