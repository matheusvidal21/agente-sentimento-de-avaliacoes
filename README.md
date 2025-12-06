# Análise de Sentimentos sob Incerteza

**Disciplina:** Introdução à Inteligência Artificial  
**Semestre:** 2025.2  
**Professor:** Andre Luis Fonseca Faustino
**Turma:** T03

## Integrantes do Grupo

- Isabela Gomes Mendes (20220038147)
- Matheus Costa Vidal (20220055246)

## Descrição do Projeto

Este projeto implementa um sistema de **análise automática de sentimentos** em avaliações de produtos escritas em português brasileiro. O sistema atua como um **agente probabilístico** que utiliza técnicas de Processamento de Linguagem Natural (PLN) e aprendizado supervisionado (Naive Bayes e Regressão Logística) para inferir o sentimento de uma avaliação como **positivo**, **neutro** ou **negativo**.

A abordagem modela o problema como inferência em ambientes com incerteza, onde o sentimento é um estado não observável que deve ser deduzido a partir de evidências textuais. O sistema inclui uma interface web interativa que permite ao usuário testar avaliações e visualizar não apenas a classificação, mas também as métricas de decisão do agente, como confiança e probabilidades por classe.

## Guia de Instalação e Execução

Siga os passos abaixo para configurar e executar o ambiente do projeto do zero.

### 1. Pré-requisitos

- **Python 3.8+** instalado no sistema
- **Git** para clonar o repositório

### 2. Instalação e Configuração do Ambiente

Clone o repositório e configure o ambiente virtual:

```bash
# Clone o repositório
git clone https://github.com/matheusvidal21/agente-sentimento-de-avaliacoes.git

# Entre na pasta do projeto
cd agente-sentimento-de-avaliacoes

# Crie um ambiente virtual (Recomendado)
python -m venv .venv

# Ative o ambiente virtual
source .venv/bin/activate  # Linux/macOS
# ou .venv\Scripts\activate # Windows

# Instale as dependências
pip install -r requirements.txt
```

### 3. Treinamento dos Modelos

Antes de executar a aplicação, é necessário treinar os modelos. O projeto já inclui um dataset pronto em `dataset/avaliacoes.csv` com 869 avaliações rotuladas.

Execute o script de treinamento:

```bash
python treinar.py
```

Este comando irá:
1. **Pré-processar** o dataset (limpeza de texto, vetorização TF-IDF)
2. **Treinar** 3 modelos de machine learning:
   - Naive Bayes (classificação de sentimento)
   - Regressão Logística (classificação de sentimento)
   - K-Means (agrupamento de perfis de usuários)
3. **Salvar** os modelos treinados na pasta `models/`:
   - `nb_modelo_sentimento.joblib`
   - `lr_modelo_sentimento.joblib`
   - `kmeans_perfil.joblib`
   - `vetorizador_tfidf.joblib`
4. **Validar** os modelos com testes manuais
5. **Gerar relatório completo** em `docs/`:
   - `relatorio_treinamento_[timestamp].md` - Métricas detalhadas, matrizes de confusão, análise dos modelos e arquitetura do sistema
   - `confusion_matrices_[timestamp].png` - Visualizações das matrizes de confusão

> **Relatório de Treinamento**: Após executar `python treinar.py`, um relatório completo em Markdown será gerado automaticamente na pasta `docs/`, contendo todas as métricas, análises e informações relevantes sobre os modelos treinados.

### 4. Execução da Aplicação Web

Após o treinamento dos modelos, execute a interface web:

```bash
streamlit run app.py
```

A aplicação estará disponível no seu navegador em: **http://localhost:8501**

## Fluxo de Arquivos e Estrutura do Projeto

### Estrutura de Diretórios

```
agente-sentimento-de-avaliacoes/
├── app.py                          # Interface web (Streamlit)
├── treinar.py                      # Script de treinamento dos modelos
├── requirements.txt                # Dependências do projeto
├── README.md                       # Documentação
├── dataset/
│   └── avaliacoes.csv             # Dataset com 869 avaliações rotuladas
├── models/                         # Modelos treinados (gerados após treinar.py)
│   ├── nb_modelo_sentimento.joblib
│   ├── lr_modelo_sentimento.joblib
│   ├── kmeans_perfil.joblib
│   └── vetorizador_tfidf.joblib
├── src/                            # Módulos do sistema
│   ├── __init__.py
│   ├── pipeline.py                # Orquestração do pipeline de treinamento
│   ├── data_preprocessing.py      # Pré-processamento e vetorização
│   ├── model_training.py          # Treinamento e avaliação dos modelos
│   ├── model_persistence.py       # Salvamento e validação dos modelos
│   ├── prediction_api.py          # API de inferência
│   └── agents/                    # Sistema multi-agente (arquitetura modular)
│       ├── __init__.py
│       ├── sentiment_agent.py     # Classificação de sentimento
│       ├── keyword_agent.py       # Extração de palavras-chave
│       ├── profiling_agent.py     # Perfilamento de clientes
│       ├── action_agent.py        # Definição de ações táticas
│       ├── response_agent.py      # Geração de respostas (LLM)
│       └── manager_agent.py       # Orquestrador do sistema
└── docs/                           # Imagens e documentação auxiliar
```

### Fluxo de Execução

#### 📊 Fluxo de Treinamento (`treinar.py`)

```
treinar.py
    ↓
pipeline.py → main()
    ↓
    ├─→ 1. data_preprocessing.py → processar_dados()
    │      • Carrega dataset/avaliacoes.csv
    │      • Limpa texto (remove acentos, caracteres especiais)
    │      • Vetoriza com TF-IDF
    │      • Divide em treino/teste (75%/25%)
    │      ↓ retorna: X_train, X_test, y_train, y_test, vectorizer
    │
    ├─→ 2. model_training.py → treinar_modelos()
    │      • Treina Naive Bayes e Regressão Logística
    │      • Treina K-Means (4 clusters)
    │      • Exibe métricas (acurácia, F1, matriz de confusão)
    │      ↓ retorna: nb_model, lr_model, kmeans_model
    │
    └─→ 3. model_persistence.py → persistir_modelos()
           • Salva modelos em models/*.joblib
           • Valida modelos com testes manuais
```

#### 🌐 Fluxo da Aplicação Web (`app.py`)

```
app.py (Streamlit)
    ↓
agents/manager_agent.py → ManagerAgent
    ↓
    ├─→ SentimentAgent (sentiment_agent.py)
    │      • Carrega modelos de models/
    │      • Classifica sentimento (positivo/neutro/negativo)
    │      • Calcula probabilidades e explica predições
    │
    ├─→ KeywordAgent (keyword_agent.py)
    │      • Extrai termos mais relevantes via TF-IDF
    │
    ├─→ ProfilingAgent (profiling_agent.py)
    │      • Identifica categoria via K-Means
    │      • Mapeia para perfis semânticos
    │
    ├─→ ActionAgent (action_agent.py)
    │      • Define ações baseadas em regras de negócio
    │
    └─→ ResponseAgent (response_agent.py)
           • Gera resposta personalizada via Gemini API
```

### Módulos Principais

| Módulo | Responsabilidade |
|--------|-----------------|
| **pipeline.py** | Orquestra o fluxo completo de treinamento |
| **data_preprocessing.py** | Limpeza de texto, vetorização TF-IDF, split de dados |
| **model_training.py** | Treinamento dos modelos (NB, LR, K-Means) e geração de métricas |
| **model_persistence.py** | Salvamento dos modelos e testes de validação |
| **prediction_api.py** | API de inferência carregando modelos persistidos |
| **agents/sentiment_agent.py** | Classificação de sentimento e explicabilidade |
| **agents/keyword_agent.py** | Extração de palavras-chave via TF-IDF |
| **agents/profiling_agent.py** | Perfilamento e categorização de clientes |
| **agents/action_agent.py** | Regras de negócio para ações táticas |
| **agents/response_agent.py** | Geração de respostas via LLM (Gemini) |
| **agents/manager_agent.py** | Orquestrador central do sistema multi-agente |
| **app.py** | Interface web interativa com Streamlit |

## Resultados e Demonstração

O sistema apresenta uma acurácia média de **~85%** (Naive Bayes) e **~86%** (Regressão Logística) no conjunto de teste.

Na interface de demonstração, o agente exibe:

1.  **Classificação do Sentimento**: Positivo, Neutro ou Negativo.
2.  **Métricas de Decisão**:
    - **Tempo de Execução**: Custo temporal da inferência.
    - **Confiança**: Grau de certeza na decisão tomada.
    - **Probabilidades Detalhadas**: Visualização da distribuição de probabilidade entre as classes possíveis.

<video width="100%" controls>
  <source src="docs/demo.mp4" type="video/mp4">
  Seu navegador não suporta a tag de vídeo.
</video>

### Métricas de Treinamento

Abaixo, as matrizes de confusão e métricas obtidas durante o treinamento dos modelos:

**Naive Bayes:**

![Treinamento Naive Bayes](docs/treinamento_naive_bayes.png)

**Regressão Logística:**

![Treinamento Regressão Logística](docs/treinamento_regressao_logistica.png)

## Referências

- **Scikit-learn**: Pedregosa et al., Scikit-learn: Machine Learning in Python, JMLR 12, pp. 2825-2830, 2011.
- **Streamlit**: Framework para criação de web apps de dados.
- **Google Generative AI**: Utilizado para geração de dados sintéticos para treinamento.
- **Naive Bayes & Logistic Regression**: Russell, S. & Norvig, P. (2010). _Artificial Intelligence: A Modern Approach_.
