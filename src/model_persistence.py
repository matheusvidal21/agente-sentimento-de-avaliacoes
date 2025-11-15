import os
import joblib


os.makedirs('models', exist_ok=True)
NB_MODEL_PATH = 'models/nb_modelo_sentimento.joblib' 
LR_MODEL_PATH = 'models/lr_modelo_sentimento.joblib'
VECTORIZER_PATH = 'models/vetorizador_tfidf.joblib'

def testar_carregamento_modelo(model_path, vectorizer_path, limpar_texto, testes_manuais):
    # Simula o carregamento dos arquivos e fazer uma previsão para garantir que tudo funcionou
    print(f"\n--- Testando o modelo salvo:  {model_path}---")

    # Carrega os arquivos
    loaded_model = joblib.load(model_path)
    loaded_vectorizer = joblib.load(vectorizer_path)

    # Mapeamento dos rótulos 
    labels = ['Negativo', 'Neutro', 'Positivo']

    # 1. Limpa os textos (usando a mesma função 'limpar_texto' do Passo 2)
    testes_limpos = [limpar_texto(texto) for texto in testes_manuais]

    # 2. Vetoriza os textos limpos (usando o vetorizador carregado)
    testes_tfidf = loaded_vectorizer.transform(testes_limpos)

    # 3. Faz a previsão (usando o modelo carregado)
    previsoes = loaded_model.predict(testes_tfidf)

    # 4. Mostra os resultados
    print("Resultados das previsões manuais:")
    for texto, previsao_numerica in zip(testes_manuais, previsoes):
        print(f"Texto: '{texto}'  ->  Previsto: {labels[previsao_numerica]}")
        
        
def persistir_modelos(nb_model, lr_model, vectorizer, limpar_texto, testes_manuais):
    """
    Salva o modelo treinado e o vetorizador TF-IDF
    em arquivos, e testa o carregamento dos mesmos.
    """
    # --- 1. Salvar os modelos ---
    
    # Salva o modelo de Naive Bayes
    joblib.dump(nb_model, NB_MODEL_PATH)
    print(f"Modelo de Naive Bayes salvo como '{NB_MODEL_PATH}'")
    
    # Salva o modelo de Regressão Logística
    joblib.dump(lr_model, LR_MODEL_PATH)
    print(f"Modelo de Regressão Logística salvo como '{LR_MODEL_PATH}'")
    
    # Salva o Vetorizador TF-IDF
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Vetorizador TF-IDF salvo como '{VECTORIZER_PATH}'")

    # --- 2. Teste de Carregamento ---
    testar_carregamento_modelo(LR_MODEL_PATH, VECTORIZER_PATH, limpar_texto, testes_manuais)
    testar_carregamento_modelo(NB_MODEL_PATH, VECTORIZER_PATH, limpar_texto, testes_manuais)
