"""
Pipeline completo de treinamento dos modelos de análise de sentimento.

Este módulo implementa o fluxo de treinamento em 3 etapas:
1. Pré-processamento: Limpeza de texto, vetorização TF-IDF
2. Treinamento: Naive Bayes, Regressão Logística, K-Means
3. Persistência: Salvamento dos modelos e validação

O pipeline utiliza o dataset com 869 avaliações de produtos,
gerando modelos prontos para inferência na aplicação web.
"""

from .data_preprocessing import processar_dados
from .model_training import treinar_modelos
from .model_persistence import persistir_modelos

DATASET_FILEPATH = "dataset/avaliacoes.csv"


def main():
    """
    Executa o pipeline completo de treinamento.
    
    Fluxo de execução:
    1. Carrega e pré-processa o dataset (limpeza, TF-IDF, split)
    2. Treina 3 modelos: NB (sentimento), LR (sentimento), K-Means (perfis)
    3. Salva modelos em disco e executa testes de validação
    """
    
    # Etapa 1: Pré-processamento dos dados
    print("--- [ETAPA 1/3] Iniciando Pré-processamento dos Dados ---")
    X_train, X_test, y_train, y_test, vectorizer, limpar_texto = processar_dados(DATASET_FILEPATH)

    # Etapa 2: Treinamento dos modelos
    print("\n--- [ETAPA 2/3] Iniciando Treinamento dos Modelos ---")
    nb_model, lr_model, kmeans_model = treinar_modelos(X_train, X_test, y_train, y_test)

    # Etapa 3: Persistência e validação
    print("\n--- [ETAPA 3/3] Iniciando Persistência e Teste ---")
    testes_manuais = [
        "A entrega demorou muito, mas o produto é bom.",
        "Tenis confortavel e bom para corrida",
        "Fantástico! Superou minhas expectativas, recomendo demais.",
        "Péssimo atendimento ao cliente, nunca mais compro aqui."
    ]
    persistir_modelos(nb_model, lr_model, kmeans_model, vectorizer, limpar_texto, testes_manuais)
    
    print("\n--- Pipeline Concluído com Sucesso! ---")


if __name__ == "__main__":
    main()