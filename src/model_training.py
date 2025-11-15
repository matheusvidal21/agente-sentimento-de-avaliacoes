import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def treinar_modelos(X_train_tfidf, X_test_tfidf, y_train, y_test):
    """
    Treina modelos de classificação de sentimento, avalia seu desempenho,
    gera visualizações e retorna o modelo especificado por 'model_to_return'.
    """
    # Mapeamento reverso para os gráficos
    labels = ['Negativo', 'Neutro', 'Positivo']

    # --- 1. Naive Bayes (MultinomialNB) ---
    print("--- 1. Treinando Naive Bayes (MultinomialNB) ---")

    # Cria e treina o modelo
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)

    # Faz as previsões nos dados de teste
    y_pred_nb = nb_model.predict(X_test_tfidf)

    # Avalia o modelo
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    f1_nb = f1_score(y_test, y_pred_nb, average='weighted')

    print(f"Acurácia (Naive Bayes): {accuracy_nb:.4f}")
    print(f"F1-Score (Naive Bayes): {f1_nb:.4f}")
    print("\nRelatório de Classificação (Naive Bayes):")
    print(classification_report(y_test, y_pred_nb, target_names=labels))

    # --- 2. Regressão Logística ---
    print("\n--- 2. Treinando Regressão Logística ---")

    # Cria e treina o modelo
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)

    # Faz as previsões nos dados de teste
    y_pred_lr = lr_model.predict(X_test_tfidf)

    # Avalia o modelo
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr, average='weighted')

    print(f"Acurácia (Regressão Logística): {accuracy_lr:.4f}")
    print(f"F1-Score (Regressão Logística): {f1_lr:.4f}")
    print("\nRelatório de Classificação (Regressão Logística):")
    print(classification_report(y_test, y_pred_lr, target_names=labels))

    # --- 3. Visualização (Matrizes de Confusão) ---
    print("\nGerando Matrizes de Confusão...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6)) # 1 linha, 2 colunas

    # Matriz para Naive Bayes
    cm_nb = confusion_matrix(y_test, y_pred_nb)
    sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title('Matriz de Confusão - Naive Bayes')
    axes[0].set_xlabel('Previsto')
    axes[0].set_ylabel('Verdadeiro')

    # Matriz para Regressão Logística
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Oranges',
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title('Matriz de Confusão - Regressão Logística')
    axes[1].set_xlabel('Previsto')
    axes[1].set_ylabel('Verdadeiro')

    plt.tight_layout()
    plt.show()
    
    return nb_model, lr_model 