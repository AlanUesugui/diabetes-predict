# 🩺 DiabetesPredict — Diagnóstico de Diabetes com Machine Learning

> Projeto Avaliativo P2 — Algoritmos de Inteligência Artificial — UNIMAR 2026

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://SEU_LINK_DO_STREAMLIT_AQUI)

---

## 👥 Integrantes e RAs

| Nome | RA |
|---|---|
| Alan Uesugui Uemura | 2084808 |
| Julio Cesar Plaza | 2046253 |


**Grupo:** 4

---

## 📋 Descrição do Problema

O diabetes mellitus é uma doença crônica que afeta milhões de pessoas no mundo. O diagnóstico precoce é fundamental para evitar complicações graves. Este projeto busca construir um modelo de classificação capaz de prever se uma paciente tem diabetes com base em medições clínicas simples, sem necessidade de exames invasivos adicionais.

---

## 🎯 Objetivo do Projeto

Desenvolver um classificador binário que, a partir de variáveis clínicas (glicose, IMC, pressão arterial, entre outras), seja capaz de prever o diagnóstico de diabetes com alta sensibilidade — priorizando a detecção de casos positivos (alto Recall) sem sacrificar completamente a precisão.

---

## 📊 Dataset Utilizado

**Pima Indians Diabetes Database**

- **Fonte:** National Institute of Diabetes and Digestive and Kidney Diseases
- **Disponível em:** [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Amostras:** 768 mulheres da população Pima Indian
- **Variáveis preditoras:** 8 (numéricas)
- **Variável-alvo:** `Outcome` (0 = Não Diabética, 1 = Diabética)
- **Desbalanceamento:** 65,1% não diabéticas / 34,9% diabéticas

### Variáveis do Dataset

| Variável | Descrição |
|---|---|
| `Pregnancies` | Número de gestações |
| `Glucose` | Concentração de glicose no plasma (mg/dL) |
| `BloodPressure` | Pressão arterial diastólica (mmHg) |
| `SkinThickness` | Espessura da dobra cutânea do tríceps (mm) |
| `Insulin` | Insulina sérica 2h após o teste (mu U/ml) |
| `BMI` | Índice de Massa Corporal (kg/m²) |
| `DiabetesPedigreeFunction` | Função que modela a história familiar de diabetes |
| `Age` | Idade em anos |
| `Outcome` | Diagnóstico (0 = Não Diabética, 1 = Diabética) — **variável-alvo** |

---

## 🤖 Tipo de Problema de Machine Learning

**Classificação binária supervisionada**

A variável-alvo `Outcome` assume dois valores: 0 (não diabética) ou 1 (diabética). O modelo aprende a partir de exemplos rotulados para prever a classe de novos casos.

---

## 🔬 Metodologia

1. **Análise Exploratória (EDA):** distribuição das classes, zeros suspeitos, outliers, correlações
2. **Pré-processamento:**
   - Zeros biologicamente impossíveis substituídos por `NaN` em 5 colunas
   - Imputação pela mediana via `SimpleImputer` dentro do Pipeline
   - Normalização via `StandardScaler`
   - Divisão estratificada 80/20 (treino/teste) com `stratify=y`
3. **Pipeline sklearn:** evita data leakage — pré-processamento fit apenas no treino
4. **Tratamento do desbalanceamento:** `class_weight='balanced'` na Regressão Logística e Random Forest
5. **Validação cruzada estratificada:** `StratifiedKFold(n_splits=5)`
6. **Otimização de hiperparâmetros:** `GridSearchCV` no Random Forest (scoring=F1)
7. **Análise estatística:** Teste de Friedman e desvio padrão por fold
8. **Avaliação final:** conjunto de teste isolado

---

## 🏋️ Modelos Treinados

| Modelo | Observação |
|---|---|
| Regressão Logística | `class_weight='balanced'`, `max_iter=1000` |
| Gaussian Naive Bayes | Sem parâmetro de balanceamento (não suportado) |
| Random Forest | `class_weight='balanced'`, `n_estimators=100` |
| **Random Forest Otimizado** | Melhor combinação encontrada pelo GridSearchCV |

---

## 🏆 Modelo Final Escolhido

**Random Forest Otimizado** (via GridSearchCV)

**Justificativa:** apresentou o melhor equilíbrio entre F1-Score e AUC-ROC no conjunto de teste, além da maior estabilidade nos folds de validação cruzada. O uso de `class_weight='balanced'` garantiu maior sensibilidade à classe minoritária (diabéticas), reduzindo Falsos Negativos — que em contexto clínico têm consequências mais graves do que alarmes falsos.

---

## 📈 Métricas de Avaliação

Métricas utilizadas: **Acurácia, Precisão, Recall, F1-Score, AUC-ROC**

> Os valores exatos das métricas são gerados dinamicamente ao executar o notebook, pois dependem dos hiperparâmetros otimizados pelo GridSearchCV em cada execução.

**Métrica prioritária:** Recall — em diagnóstico clínico, minimizar Falsos Negativos (diabetes não detectada) é mais importante do que minimizar alarmes falsos.

**Análise estatística:** Teste de Friedman aplicado sobre os 5 folds para verificar se as diferenças de desempenho entre modelos são estatisticamente significativas.

---

## 🏗️ Estrutura dos Arquivos

```
diabetes-predict/
│
├── app.py                          # Aplicação Streamlit
├── requirements.txt                # Dependências do projeto
├── README.md                       # Esta documentação
│
├── notebooks/
│   └── notebook_atualizado.ipynb   # Notebook revisado (P2)
│
├── model/
│   └── modelo_final.joblib         # Modelo final salvo (Pipeline completo)
│   └── colunas.joblib              # Colunas de entrada do modelo
│
├── reports/
│   └── relatorio_atualizado.pdf    # Relatório final em PDF
│
└── data/
    └── diabetes.csv                # Dataset utilizado
```

---

## 🛠️ Tecnologias Utilizadas

| Tecnologia | Uso |
|---|---|
| Python 3.10+ | Linguagem principal |
| pandas | Manipulação de dados |
| numpy | Operações numéricas |
| matplotlib / seaborn | Visualizações |
| scikit-learn | Modelagem, pipeline e métricas |
| scipy | Teste estatístico de Friedman |
| joblib | Salvamento e carregamento do modelo |
| Streamlit | Aplicação web interativa |

---

## ▶️ Como Executar o Notebook

```bash
# 1. Clone o repositório
git clone https://github.com/SEU_USUARIO/diabetes-predict.git
cd diabetes-predict

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Certifique-se de que o arquivo diabetes.csv está em data/
# (ou na raiz, conforme o caminho usado no notebook)

# 4. Abra o notebook
jupyter notebook notebooks/notebook_atualizado.ipynb

# 5. Execute todas as células em ordem (Kernel > Restart & Run All)
# O modelo será salvo automaticamente em model/modelo_final.joblib
```

---

## ▶️ Como Executar o App Streamlit Localmente

```bash
# 1. Certifique-se de que o modelo já foi gerado pelo notebook
# (arquivo model/modelo_final.joblib deve existir)

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Execute o app
streamlit run app.py
```

O app abrirá automaticamente no navegador em `http://localhost:8501`.

---

## 🌐 Link do App Publicado

**[Acesse o app aqui](https://SEU_LINK_DO_STREAMLIT_AQUI)**

> Deploy realizado via [Streamlit Community Cloud](https://share.streamlit.io)

---

## ⚠️ Limitações

- O dataset é relativamente pequeno (768 amostras), o que limita a capacidade de generalização do modelo
- A população Pima Indian tem características genéticas e metabólicas específicas — o modelo pode não generalizar bem para outras populações
- Variáveis com alto percentual de zeros suspeitos (Insulin: 48,7%, SkinThickness: 29,6%) introduzem incerteza mesmo após imputação pela mediana
- O modelo não substitui avaliação médica profissional — deve ser usado apenas como ferramenta de triagem de apoio

---

## ✅ Conclusão

O projeto demonstrou que é possível construir um classificador confiável para diagnóstico de diabetes utilizando dados clínicos simples. O Random Forest Otimizado com `class_weight='balanced'` e otimização via GridSearchCV apresentou o melhor desempenho geral, com bom equilíbrio entre Precisão e Recall.

A análise estatística via Teste de Friedman permitiu investigar formalmente se as diferenças entre os modelos são significativas — indo além da comparação visual e atendendo à exigência de maior profundidade analítica.

A aplicação Streamlit permite que qualquer usuário insira dados clínicos e obtenha uma predição imediata, demonstrando o potencial de uso prático do modelo treinado.

---

*Projeto desenvolvido para fins acadêmicos — UNIMAR 2026*
