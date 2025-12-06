"""
Agente de Análise de Sentimento.

Responsável por classificar o sentimento de textos usando modelos
de Machine Learning (Naive Bayes ou Regressão Logística).
"""

from typing import Dict, List, Tuple
from ..data_preprocessing import limpar_texto


class SentimentAgent:
    """
    Agente especializado em classificação de sentimento.
    
    Utiliza modelos supervisionados (NB/LR) treinados com TF-IDF
    para inferir se uma avaliação é Positiva, Neutra ou Negativa.
    """
    
    def __init__(self, model, vectorizer):
        """
        Inicializa o agente de sentimento.
        
        Args:
            model: Modelo scikit-learn treinado (MultinomialNB ou LogisticRegression)
            vectorizer: TfidfVectorizer treinado no corpus
        """
        self.model = model
        self.vectorizer = vectorizer
        self.labels = ["Negativo", "Neutro", "Positivo"]

    def predict(self, text: str) -> Dict[str, any]:
        """
        Classifica o sentimento de um texto.
        
        Args:
            text: Texto da avaliação a ser classificada
            
        Returns:
            Dicionário com label e probabilidades por classe
        """
        texto_limpo = limpar_texto(text)
        texto_tfidf = self.vectorizer.transform([texto_limpo])
        
        previsao_numerica = self.model.predict(texto_tfidf)[0]
        probabilidades = self.model.predict_proba(texto_tfidf)[0]

        return {
            "label": self.labels[previsao_numerica],
            "probabilities": {
                label: prob 
                for label, prob in zip(self.labels, probabilidades)
            },
        }

    def explain(self, text: str) -> List[Tuple[str, float]]:
        """
        Explica a predição mostrando contribuição de cada palavra.
        
        Retorna lista de tuplas (palavra, score) indicando quanto cada
        termo contribuiu para a classificação. Funciona melhor com LR.
        
        Args:
            text: Texto a ser explicado
            
        Returns:
            Lista de tuplas (palavra, score) ordenadas por contribuição
        """
        texto_limpo = limpar_texto(text)
        texto_tfidf = self.vectorizer.transform([texto_limpo])
        feature_names = self.vectorizer.get_feature_names_out()
        
        if not hasattr(self.model, "coef_") and not hasattr(self.model, "feature_log_prob_"):
            return []

        try:
            if hasattr(self.model, "coef_"):
                # Regressão Logística: Diferença entre coeficientes Positivo-Negativo
                if self.model.coef_.shape[0] == 1:
                    coefs = self.model.coef_[0]
                else:
                    coefs = self.model.coef_[2] - self.model.coef_[0]
            elif hasattr(self.model, "feature_log_prob_"):
                # Naive Bayes: Diferença entre log-probabilidades
                coefs = self.model.feature_log_prob_[2] - self.model.feature_log_prob_[0]
            else:
                return []

            rows, cols = texto_tfidf.nonzero()
            
            explanation = []
            for col in cols:
                word = feature_names[col]
                score = coefs[col] * texto_tfidf[0, col]
                explanation.append((word, score))
            
            explanation.sort(key=lambda x: abs(x[1]), reverse=True)
            return explanation

        except Exception as e:
            print(f"Erro ao explicar: {e}")
            return []
