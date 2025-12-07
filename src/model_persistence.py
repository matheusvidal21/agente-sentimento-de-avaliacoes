"""
Módulo de Persistência de Modelos.

Responsável por salvar modelos treinados em disco e validar
o carregamento através de testes de inferência.
"""

import os
import joblib


os.makedirs('models', exist_ok=True)

NB_MODEL_PATH = 'models/nb_modelo_sentimento.joblib' 
LR_MODEL_PATH = 'models/lr_modelo_sentimento.joblib'
VECTORIZER_PATH = 'models/vetorizador_tfidf.joblib'


def testar_carregamento_modelo(
    model_path: str, 
    vectorizer_path: str, 
    limpar_texto, 
    testes_manuais: list
) -> None:
    """
    Valida modelos salvos realizando inferências de teste.
    
    Args:
        model_path: Caminho do arquivo .joblib do modelo
        vectorizer_path: Caminho do vetorizador TF-IDF
        limpar_texto: Função de limpeza de texto
        testes_manuais: Lista de textos para teste
    """
    print(f"\n--- Testando modelo: {model_path} ---")

    loaded_model = joblib.load(model_path)
    loaded_vectorizer = joblib.load(vectorizer_path)

    labels = ['Negativo', 'Neutro', 'Positivo']

    testes_limpos = [limpar_texto(texto) for texto in testes_manuais]
    testes_tfidf = loaded_vectorizer.transform(testes_limpos)
    previsoes = loaded_model.predict(testes_tfidf)

    print(f"Resultados das previsões (Sentimento):")
    for texto, previsao_numerica in zip(testes_manuais, previsoes):
        print(f"'{texto}' → {labels[previsao_numerica]}")
        
        
def persistir_modelos(
    nb_model, 
    lr_model, 
    vectorizer, 
    limpar_texto, 
    testes_manuais: list
) -> None:
    """
    Serializa modelos treinados e executa testes de validação.
    
    Salva todos os artefatos em formato .joblib e verifica a integridade
    através de predições em textos de exemplo.
    
    Args:
        nb_model: Modelo Naive Bayes treinado
        lr_model: Modelo Regressão Logística treinado
        vectorizer: Vetorizador TF-IDF treinado
        limpar_texto: Função de limpeza de texto
        testes_manuais: Lista de textos para validação
    """
    joblib.dump(nb_model, NB_MODEL_PATH)
    print(f"Naive Bayes salvo: '{NB_MODEL_PATH}'")
    
    joblib.dump(lr_model, LR_MODEL_PATH)
    print(f"Regressão Logística salva: '{LR_MODEL_PATH}'")
    
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Vetorizador TF-IDF salvo: '{VECTORIZER_PATH}'")

    testar_carregamento_modelo(LR_MODEL_PATH, VECTORIZER_PATH, limpar_texto, testes_manuais)
    testar_carregamento_modelo(NB_MODEL_PATH, VECTORIZER_PATH, limpar_texto, testes_manuais)
