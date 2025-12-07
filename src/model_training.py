"""
Módulo de Treinamento de Modelos.

Responsável por treinar os modelos de classificação (Naive Bayes
e Regressão Logística) e gerar métricas e visualizações de desempenho.
"""

import os
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_score, recall_score


os.makedirs('docs', exist_ok=True)


def treinar_modelos(X_train_tfidf, X_test_tfidf, y_train, y_test) -> tuple:
    """
    Treina modelos de ML e gera visualizações de performance.
    
    Treina 2 modelos supervisionados: Naive Bayes e Regressão Logística
    para classificação de sentimento.
    
    Args:
        X_train_tfidf: Matriz TF-IDF de treino
        X_test_tfidf: Matriz TF-IDF de teste
        y_train: Labels de treino
        y_test: Labels de teste
        
    Returns:
        Tupla (nb_model, lr_model)
    """
    labels = ['Negativo', 'Neutro', 'Positivo']

    # Naive Bayes
    print("--- 1. Treinando Naive Bayes (MultinomialNB) ---")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    y_pred_nb = nb_model.predict(X_test_tfidf)

    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    f1_nb = f1_score(y_test, y_pred_nb, average='weighted')

    print(f"Acurácia (Naive Bayes): {accuracy_nb:.4f}")
    print(f"F1-Score (Naive Bayes): {f1_nb:.4f}")
    print("\nRelatório de Classificação (Naive Bayes):")
    print(classification_report(y_test, y_pred_nb, target_names=labels))

    # Regressão Logística
    print("\n--- 2. Treinando Regressão Logística ---")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)
    y_pred_lr = lr_model.predict(X_test_tfidf)

    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr, average='weighted')

    print(f"Acurácia (Regressão Logística): {accuracy_lr:.4f}")
    print(f"F1-Score (Regressão Logística): {f1_lr:.4f}")
    print("\nRelatório de Classificação (Regressão Logística):")
    print(classification_report(y_test, y_pred_lr, target_names=labels))

    # Matrizes de Confusão
    print("\nGerando Matrizes de Confusão...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cm_nb = confusion_matrix(y_test, y_pred_nb)
    sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title('Matriz de Confusão - Naive Bayes')
    axes[0].set_xlabel('Previsto')
    axes[0].set_ylabel('Verdadeiro')

    cm_lr = confusion_matrix(y_test, y_pred_lr)
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Oranges',
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title('Matriz de Confusão - Regressão Logística')
    axes[1].set_xlabel('Previsto')
    axes[1].set_ylabel('Verdadeiro')
    
    plt.tight_layout()
    
    # Salvar figura
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    confusion_matrix_path = f'docs/confusion_matrices_{timestamp}.png'
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    print(f"Matrizes de confusão salvas em: {confusion_matrix_path}")
    plt.close()
    
    # Gerar relatório de treinamento
    _gerar_relatorio_treinamento(
        nb_model, lr_model,
        y_test, y_pred_nb, y_pred_lr,
        accuracy_nb, accuracy_lr,
        f1_nb, f1_lr,
        cm_nb, cm_lr,
        labels,
        X_train_tfidf, X_test_tfidf,
        y_train,
        confusion_matrix_path,
        timestamp
    )
    
    return nb_model, lr_model


def _gerar_relatorio_treinamento(
    nb_model, lr_model,
    y_test, y_pred_nb, y_pred_lr,
    accuracy_nb, accuracy_lr,
    f1_nb, f1_lr,
    cm_nb, cm_lr,
    labels,
    X_train_tfidf, X_test_tfidf,
    y_train,
    confusion_matrix_path,
    timestamp
) -> None:
    """
    Gera relatório completo de treinamento em Markdown.
    
    Cria documento em docs/ com todas as métricas, matrizes de confusão,
    informações sobre os modelos e arquitetura do sistema.
    """
    report_path = f'docs/relatorio_treinamento_{timestamp}.md'
    
    # Calcular métricas adicionais
    precision_nb = precision_score(y_test, y_pred_nb, average='weighted')
    recall_nb = recall_score(y_test, y_pred_nb, average='weighted')
    precision_lr = precision_score(y_test, y_pred_lr, average='weighted')
    recall_lr = recall_score(y_test, y_pred_lr, average='weighted')
    
    # Distribuição de classes
    train_dist = y_train.value_counts(normalize=True).sort_index()
    test_dist = y_test.value_counts(normalize=True).sort_index()
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"""# Relatório de Treinamento - Sistema de Análise de Sentimentos

**Data de Treinamento:** {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}  
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
- **Total de Amostras:** {len(y_train) + len(y_test)}
- **Conjunto de Treino:** {len(y_train)} amostras ({len(y_train)/(len(y_train) + len(y_test))*100:.1f}%)
- **Conjunto de Teste:** {len(y_test)} amostras ({len(y_test)/(len(y_train) + len(y_test))*100:.1f}%)
- **Dimensionalidade TF-IDF:** {X_train_tfidf.shape[1]} features

### Distribuição de Classes

#### Conjunto de Treino
| Classe | Quantidade | Proporção |
|--------|-----------|-----------|
| Negativo | {(y_train == 0).sum()} | {train_dist[0]*100:.2f}% |
| Neutro | {(y_train == 1).sum()} | {train_dist[1]*100:.2f}% |
| Positivo | {(y_train == 2).sum()} | {train_dist[2]*100:.2f}% |

#### Conjunto de Teste
| Classe | Quantidade | Proporção |
|--------|-----------|-----------|
| Negativo | {(y_test == 0).sum()} | {test_dist[0]*100:.2f}% |
| Neutro | {(y_test == 1).sum()} | {test_dist[1]*100:.2f}% |
| Positivo | {(y_test == 2).sum()} | {test_dist[2]*100:.2f}% |

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
| **Acurácia** | {accuracy_nb*100:.2f}% |
| **Precisão (Weighted)** | {precision_nb*100:.2f}% |
| **Recall (Weighted)** | {recall_nb*100:.2f}% |
| **F1-Score (Weighted)** | {f1_nb*100:.2f}% |

### Relatório Detalhado por Classe

```
{classification_report(y_test, y_pred_nb, target_names=labels)}
```

### Matriz de Confusão

```
              Previsto
            Neg  Neu  Pos
Verdadeiro
    Neg     {cm_nb[0][0]:3d}  {cm_nb[0][1]:3d}  {cm_nb[0][2]:3d}
    Neu     {cm_nb[1][0]:3d}  {cm_nb[1][1]:3d}  {cm_nb[1][2]:3d}
    Pos     {cm_nb[2][0]:3d}  {cm_nb[2][1]:3d}  {cm_nb[2][2]:3d}
```

### Interpretação
- **Pontos Fortes:** {_interpretar_modelo_nb(cm_nb, accuracy_nb)}
- **Desafios:** {_identificar_desafios_nb(cm_nb)}

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
| **Acurácia** | {accuracy_lr*100:.2f}% |
| **Precisão (Weighted)** | {precision_lr*100:.2f}% |
| **Recall (Weighted)** | {recall_lr*100:.2f}% |
| **F1-Score (Weighted)** | {f1_lr*100:.2f}% |

### Relatório Detalhado por Classe

```
{classification_report(y_test, y_pred_lr, target_names=labels)}
```

### Matriz de Confusão

```
              Previsto
            Neg  Neu  Pos
Verdadeiro
    Neg     {cm_lr[0][0]:3d}  {cm_lr[0][1]:3d}  {cm_lr[0][2]:3d}
    Neu     {cm_lr[1][0]:3d}  {cm_lr[1][1]:3d}  {cm_lr[1][2]:3d}
    Pos     {cm_lr[2][0]:3d}  {cm_lr[2][1]:3d}  {cm_lr[2][2]:3d}
```

### Interpretação
- **Pontos Fortes:** {_interpretar_modelo_lr(cm_lr, accuracy_lr)}
- **Desafios:** {_identificar_desafios_lr(cm_lr)}

---

## Visualizações

![Matrizes de Confusão]({os.path.basename(confusion_matrix_path)})

As matrizes de confusão acima mostram a comparação visual entre os dois modelos de classificação. Células na diagonal representam predições corretas.

---

## Arquitetura do Sistema Multi-Agente

### Agentes Especializados

#### 1. SentimentAgent
- **Função:** Classificação de sentimento (Positivo/Neutro/Negativo)
- **Modelos:** Naive Bayes e Regressão Logística
- **Features:** Explica contribuição de cada palavra para a predição

#### 2. ValidationAgent
- **Função:** Quantificação de incerteza e arbitragem entre modelos
- **Propósito:** Validar predições e escolher o melhor modelo para cada caso

#### 3. KeywordAgent
- **Função:** Extração de palavras-chave via TF-IDF
- **Propósito:** Identificar termos mais relevantes da avaliação

#### 4. ActionAgent
- **Função:** Definição de ações táticas baseadas em regras
- **Propósito:** Recomendar próximos passos com base no contexto

#### 5. ResponseAgent
- **Função:** Geração de respostas automáticas via LLM (Gemini)
- **Propósito:** Criar respostas empáticas e contextualizadas

#### 6. ManagerAgent
- **Função:** Orquestração de todos os agentes
- **Propósito:** Coordenar o pipeline completo de análise

### Pipeline de Execução

```
Texto do Cliente
    ↓
[SentimentAgent] → Classifica sentimento (NB + LR)
    ↓
[ValidationAgent] → Valida e escolhe melhor modelo
    ↓
[KeywordAgent] → Extrai termos-chave
    ↓
[ActionAgent] → Define ação tática
    ↓
[ResponseAgent] → Gera resposta personalizada
    ↓
Resultado Consolidado
```

---

## Comparação de Modelos

| Métrica | Naive Bayes | Regressão Logística | Melhor |
|---------|-------------|---------------------|--------|
| Acurácia | {accuracy_nb*100:.2f}% | {accuracy_lr*100:.2f}% | {'LR' if accuracy_lr > accuracy_nb else 'NB' if accuracy_nb > accuracy_lr else 'Empate'} |
| Precisão | {precision_nb*100:.2f}% | {precision_lr*100:.2f}% | {'LR' if precision_lr > precision_nb else 'NB' if precision_nb > precision_lr else 'Empate'} |
| Recall | {recall_nb*100:.2f}% | {recall_lr*100:.2f}% | {'LR' if recall_lr > recall_nb else 'NB' if recall_nb > recall_lr else 'Empate'} |
| F1-Score | {f1_nb*100:.2f}% | {f1_lr*100:.2f}% | {'LR' if f1_lr > f1_nb else 'NB' if f1_nb > f1_lr else 'Empate'} |

### Recomendação
{'**Regressão Logística**' if accuracy_lr >= accuracy_nb else '**Naive Bayes**'} apresentou melhor desempenho geral e é o modelo padrão na interface web.

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
| `docs/{os.path.basename(confusion_matrix_path)}` | Matrizes de confusão (PNG) |
| `docs/{os.path.basename(report_path)}` | Este relatório |

---

## Referências

1. **Scikit-learn:** Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, pp. 2825-2830, 2011.
2. **Naive Bayes:** McCallum, A., & Nigam, K. (1998). *A comparison of event models for naive bayes text classification.*
3. **TF-IDF:** Salton, G., & Buckley, C. (1988). *Term-weighting approaches in automatic text retrieval.*

---

**Relatório gerado automaticamente pelo pipeline de treinamento.**
""")
    
    print(f"\nRelatório de treinamento salvo em: {report_path}")


def _interpretar_modelo_nb(cm, accuracy):
    """Gera interpretação dos pontos fortes do Naive Bayes."""
    if accuracy > 0.85:
        return "Excelente desempenho geral, especialmente em classificação de extremos (Positivo/Negativo)"
    elif accuracy > 0.75:
        return "Bom desempenho, modelo robusto para classificação de texto"
    else:
        return "Desempenho adequado, pode se beneficiar de mais dados de treinamento"


def _identificar_desafios_nb(cm):
    """Identifica principais desafios do Naive Bayes."""
    # Confusão entre Neutro e outras classes
    neutro_confusao = cm[1][0] + cm[1][2]
    if neutro_confusao > cm[1][1] * 0.3:
        return "Maior dificuldade em classificar avaliações Neutras (confusão com extremos)"
    return "Boa separação entre classes, baixa confusão"


def _interpretar_modelo_lr(cm, accuracy):
    """Gera interpretação dos pontos fortes da Regressão Logística."""
    if accuracy > 0.85:
        return "Excelente capacidade de separação linear, bom uso de features TF-IDF"
    elif accuracy > 0.75:
        return "Bom ajuste aos dados, coeficientes permitem explicabilidade"
    else:
        return "Performance adequada, considerar feature engineering adicional"


def _identificar_desafios_lr(cm):
    """Identifica principais desafios da Regressão Logística."""
    neutro_confusao = cm[1][0] + cm[1][2]
    if neutro_confusao > cm[1][1] * 0.3:
        return "Classe Neutro apresenta mais confusão (característica comum em sentimento ternário)"
    return "Classificação balanceada entre todas as classes" 