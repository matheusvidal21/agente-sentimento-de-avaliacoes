from .dataset_generator import gerar_dataset, DATASET_FILEPATH
from .data_preprocessing import processar_dados
from .model_training import treinar_modelos
from .model_persistence import persistir_modelos


def main():
    print("--- [ETAPA 1/4] Geração do Dataset Concluída ---")
    gerar_dataset()

    print("\n--- [ETAPA 2/4] Iniciando Pré-processamento dos Dados ---")
    X_train, X_test, y_train, y_test, vectorizer, limpar_texto = processar_dados(DATASET_FILEPATH)

    print("\n--- [ETAPA 3/4] Iniciando Treinamento dos Modelos ---")
    nb_model, lr_model = treinar_modelos(X_train, X_test, y_train, y_test)

    print("\n--- [ETAPA 4/4] Iniciando Persistência e Teste ---")
    # --- Crie suas próprias frases de teste aqui ---
    testes_manuais = [
        "A entrega demorou muito, mas o produto é bom.", # Neutro
        "Tenis confortavel e bom para corrida", # Positivo
        "Fantástico! Superou minhas expectativas, recomendo demais.", # Positivo
        "Péssimo atendimento ao cliente, nunca mais compro aqui." # Negativo
    ]
    persistir_modelos(nb_model, lr_model, vectorizer, limpar_texto, testes_manuais)
    
    print("\n--- Pipeline Concluído com Sucesso! ---")


if __name__ == "__main__":
    main()