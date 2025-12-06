"""
API de Predição de Sentimento.

Módulo simplificado para inferência direta de sentimentos,
usado pela interface Streamlit com cache de artefatos.
"""

import joblib
import streamlit as st
from typing import Dict, Optional, Tuple
from .model_persistence import NB_MODEL_PATH, LR_MODEL_PATH, VECTORIZER_PATH


LABELS = ["Negativo", "Neutro", "Positivo"]


@st.cache_resource
def load_artifacts(model_type: str = "lr") -> Tuple[Optional[object], Optional[object]]:
    """
    Carrega modelo e vetorizador com cache do Streamlit.
    
    Args:
        model_type: Tipo do modelo ("nb" ou "lr")
        
    Returns:
        Tupla (model, vectorizer) ou (None, None) em caso de erro
    """
    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
        model = joblib.load(NB_MODEL_PATH if model_type == "nb" else LR_MODEL_PATH)
        return model, vectorizer
    except FileNotFoundError as e:
        print(f"Erro ao carregar artefatos: {e}")
        return None, None


def prever_sentimento(texto: str, model_type: str, limpar_texto_func) -> Dict[str, any]:
    """
    Executa inferência de sentimento em um texto.
    
    Args:
        texto: Texto da avaliação
        model_type: Tipo do modelo ("nb" ou "lr")
        limpar_texto_func: Função de limpeza de texto
        
    Returns:
        Dicionário com label e probabilidades
    """
    model, vectorizer = load_artifacts(model_type)

    if model is None:
        return "Erro: Modelos não carregados. Execute o pipeline de treino primeiro."

    texto_limpo = limpar_texto_func(texto)
    texto_tfidf = vectorizer.transform([texto_limpo])

    previsao_numerica = model.predict(texto_tfidf)[0]
    probabilidades = model.predict_proba(texto_tfidf)[0]

    return {
        "label": LABELS[previsao_numerica],
        "probabilities": {label: prob for label, prob in zip(LABELS, probabilidades)},
    }
