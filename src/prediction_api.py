import joblib
import streamlit as st
from .model_persistence import NB_MODEL_PATH, LR_MODEL_PATH, VECTORIZER_PATH


# Mapeamento reverso para os rótulos de saída
LABELS = ['Negativo', 'Neutro', 'Positivo']


@st.cache_resource # Uso de cache do Streamlit para não carregar o modelo toda vez
def load_artifacts(model_type='lr'):
    """Carrega o modelo e o vetorizador na memória."""
    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
        
        if model_type == 'nb':
            model = joblib.load(NB_MODEL_PATH)
        else: # Padrão é LR
            model = joblib.load(LR_MODEL_PATH)

        return model, vectorizer

    except FileNotFoundError as e:
        print(f"Erro ao carregar o artefato: {e}")
        return None, None
    

def prever_sentimento(texto, model_type, limpar_texto_func):
    """Executa a inferência para um único texto."""
    model, vectorizer = load_artifacts(model_type)

    if model is None:
        return "Erro: Modelos não carregados. Execute o pipeline de treino primeiro."

    # 1. Limpeza
    texto_limpo = limpar_texto_func(texto)

    # 2. Vetorização
    texto_tfidf = vectorizer.transform([texto_limpo])

    # 3. Previsão
    previsao_numerica = model.predict(texto_tfidf)[0]

    return LABELS[previsao_numerica]