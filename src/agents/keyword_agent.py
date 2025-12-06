"""
Agente de Extração de Palavras-Chave.

Responsável por identificar os termos mais relevantes em uma avaliação
usando valores TF-IDF.
"""

from typing import List
from ..data_preprocessing import limpar_texto


class KeywordAgent:
    """
    Agente especializado em extração de palavras-chave.
    
    Utiliza TF-IDF para identificar os termos mais representativos
    de uma avaliação no contexto do corpus completo.
    """
    
    def __init__(self, vectorizer):
        """
        Inicializa o agente de keywords.
        
        Args:
            vectorizer: TfidfVectorizer treinado no corpus
        """
        self.vectorizer = vectorizer

    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """
        Extrai as palavras-chave mais relevantes de um texto.
        
        Args:
            text: Texto da avaliação
            top_n: Quantidade de keywords a retornar
            
        Returns:
            Lista com as top_n palavras mais relevantes
        """
        texto_limpo = limpar_texto(text)
        tfidf_matrix = self.vectorizer.transform([texto_limpo])
        
        feature_names = self.vectorizer.get_feature_names_out()
        row = tfidf_matrix[0]
        
        if row.nnz == 0:
            return []
        
        tuples = zip(row.data, row.indices)
        sorted_tuples = sorted(tuples, key=lambda x: x[0], reverse=True)
        top_tuples = sorted_tuples[:top_n]
        
        keywords = [feature_names[idx] for score, idx in top_tuples]
        
        return keywords
