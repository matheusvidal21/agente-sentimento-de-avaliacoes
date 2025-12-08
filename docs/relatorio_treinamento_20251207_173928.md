# Relatório de Treinamento - Sistema de Análise de Sentimentos

**Data de Treinamento:** 07/12/2025 17:39:28  
**Disciplina:** Introdução à Inteligência Artificial  
**Semestre:** 2025.2

---

## Sumário Executivo

Este relatório documenta o processo de treinamento dos modelos de Machine Learning utilizados no Sistema Multi-Agente para Análise de Sentimentos. O sistema implementa uma arquitetura baseada em agentes especializados que trabalham em conjunto para classificar sentimentos e gerar respostas automáticas.

### Modelos Treinados
1. **Naive Bayes (MultinomialNB)** - Classificação de Sentimento
2. **Regressão Logística** - Classificação de Sentimento

---

## Dataset

### Características
- **Total de Amostras:** 869
- **Conjunto de Treino:** 651 amostras (74.9%)
- **Conjunto de Teste:** 218 amostras (25.1%)
- **Dimensionalidade TF-IDF:** 820 features

### Distribuição de Classes

#### Conjunto de Treino
| Classe | Quantidade | Proporção |
|--------|-----------|-----------|
| Negativo | 240 | 36.87% |
| Neutro | 149 | 22.89% |
| Positivo | 262 | 40.25% |

#### Conjunto de Teste
| Classe | Quantidade | Proporção |
|--------|-----------|-----------|
| Negativo | 80 | 36.70% |
| Neutro | 50 | 22.94% |
| Positivo | 88 | 40.37% |

**Observação:** Dataset estratificado - distribuição proporcional mantida entre treino e teste.

---

## Modelo 1: Naive Bayes (MultinomialNB)

### Hiperparâmetros
- **Algoritmo:** Multinomial Naive Bayes
- **Prior:** Uniforme (aprendido dos dados)
- **Alpha (Suavização de Laplace):** 1.0 (padrão)

### Métricas de Performance

| Métrica | Valor |
|---------|-------|
| **Acurácia** | 93.58% |
| **Precisão (Weighted)** | 94.04% |
| **Recall (Weighted)** | 93.58% |
| **F1-Score (Weighted)** | 93.47% |

### Relatório Detalhado por Classe

```
              precision    recall  f1-score   support

    Negativo       0.89      0.97      0.93        80
      Neutro       1.00      0.80      0.89        50
    Positivo       0.96      0.98      0.97        88

    accuracy                           0.94       218
   macro avg       0.95      0.92      0.93       218
weighted avg       0.94      0.94      0.93       218

```

### Matriz de Confusão

```
              Previsto
            Neg  Neu  Pos
Verdadeiro
    Neg      78    0    2
    Neu       8   40    2
    Pos       2    0   86
```

### Interpretação
- **Pontos Fortes:** Excelente desempenho geral, especialmente em classificação de extremos (Positivo/Negativo)
- **Desafios:** Boa separação entre classes, baixa confusão

---

## Modelo 2: Regressão Logística

### Hiperparâmetros
- **Solver:** lbfgs
- **Max Iterações:** 1000
- **Regularização:** L2 (padrão)
- **Random State:** 42
- **Multi-class:** Multinomial (One-vs-Rest)

### Métricas de Performance

| Métrica | Valor |
|---------|-------|
| **Acurácia** | 90.37% |
| **Precisão (Weighted)** | 90.95% |
| **Recall (Weighted)** | 90.37% |
| **F1-Score (Weighted)** | 90.11% |

### Relatório Detalhado por Classe

```
              precision    recall  f1-score   support

    Negativo       0.85      0.95      0.90        80
      Neutro       0.97      0.72      0.83        50
    Positivo       0.92      0.97      0.94        88

    accuracy                           0.90       218
   macro avg       0.92      0.88      0.89       218
weighted avg       0.91      0.90      0.90       218

```

### Matriz de Confusão

```
              Previsto
            Neg  Neu  Pos
Verdadeiro
    Neg      76    0    4
    Neu      11   36    3
    Pos       2    1   85
```

### Interpretação
- **Pontos Fortes:** Excelente capacidade de separação linear, bom uso de features TF-IDF
- **Desafios:** Classe Neutro apresenta mais confusão (característica comum em sentimento ternário)

---

## Visualizações

![Matrizes de Confusão](confusion_matrices_20251207_173928.png)

As matrizes de confusão acima mostram a comparação visual entre os dois modelos de classificação. Células na diagonal representam predições corretas.

---

## Comparação de Modelos

| Métrica | Naive Bayes | Regressão Logística | Melhor |
|---------|-------------|---------------------|--------|
| Acurácia | 93.58% | 90.37% | NB |
| Precisão | 94.04% | 90.95% | NB |
| Recall | 93.58% | 90.37% | NB |
| F1-Score | 93.47% | 90.11% | NB |

### Recomendação
**Naive Bayes** apresentou melhor desempenho geral e é o modelo padrão na interface web.

---

## Considerações Técnicas

### Pré-processamento
1. **Limpeza:** Remoção de acentos, caracteres especiais, normalização
2. **Tokenização:** Unigramas e bigramas (n-grams 1-2)
3. **Vetorização:** TF-IDF com stopwords em português
4. **Filtro:** Termos presentes em menos de 3 documentos removidos

### Validação
- **Estratégia:** Holdout estratificado (75% treino, 25% teste)
- **Métrica Principal:** F1-Score (balanceia precisão e recall)
- **Testes Manuais:** Validação qualitativa com textos diversos

### Explicabilidade
A Regressão Logística permite explicar predições através dos coeficientes, mostrando quais palavras contribuem positiva ou negativamente para cada classe.

---

## Artefatos Gerados

| Arquivo | Descrição |
|---------|-----------|
| `models/nb_modelo_sentimento.joblib` | Modelo Naive Bayes serializado |
| `models/lr_modelo_sentimento.joblib` | Modelo Regressão Logística serializado |
| `models/vetorizador_tfidf.joblib` | Vetorizador TF-IDF treinado |
| `docs/confusion_matrices_20251207_173928.png` | Matrizes de confusão (PNG) |
| `docs/relatorio_treinamento_20251207_173928.md` | Este relatório |

---

## Referências

1. **Scikit-learn:** Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, pp. 2825-2830, 2011.
2. **Naive Bayes:** McCallum, A., & Nigam, K. (1998). *A comparison of event models for naive bayes text classification.*
3. **TF-IDF:** Salton, G., & Buckley, C. (1988). *Term-weighting approaches in automatic text retrieval.*

---

**Relatório gerado automaticamente pelo pipeline de treinamento.**
