"""
Agente de Extração de Palavras-Chave.

Responsável por identificar os termos mais relevantes em uma avaliação
usando valores TF-IDF.

Especificação PEAS:
    Performance: Extrair termos semanticamente relevantes; Evitar ruído
    Environment: Matriz TF-IDF; Vocabulário do corpus; Texto original
    Actuators: Emitir lista de keywords; Destacar termos; Sinalizar OOV
    Sensors: Receber texto; Observar scores TF-IDF; Detectar termos ausentes

Fundamentação Teórica:
    TF-IDF (Term Frequency-Inverse Document Frequency) é uma medida
    estatística que avalia a importância de um termo em um documento
    dentro de um corpus. Este agente utiliza essa técnica para
    identificar palavras-chave discriminativas.
    
    Referências:
    - Russell, S. & Norvig, P. (2020). Artificial Intelligence: A Modern Approach
"""

from typing import List, Dict, Any, Optional, Tuple
from .base_agent import BaseAgent, PEAS, AgentPercept
from ..data_preprocessing import limpar_texto


class KeywordAgent(BaseAgent):
    """
    Agente especializado em extração de palavras-chave.
    
    Utiliza TF-IDF para identificar os termos mais representativos
    de uma avaliação no contexto do corpus completo.
    
    Comportamento autônomo:
    - Filtra termos com baixo score TF-IDF
    - Alerta quando texto não contém termos do vocabulário
    - Adapta quantidade de keywords ao tamanho do texto
    
    Attributes:
        vectorizer: TfidfVectorizer treinado no corpus
        MIN_TFIDF_SCORE: Score mínimo para considerar um termo relevante
    """
    
    MIN_TFIDF_SCORE = 0.1  # Score mínimo para considerar relevante
    
    def __init__(self, vectorizer, name: str = "KeywordAgent"):
        """
        Inicializa o agente de keywords.
        
        Args:
            vectorizer: TfidfVectorizer treinado no corpus
            name: Identificador do agente
        """
        super().__init__(name)
        self.vectorizer = vectorizer
        
        # Objetivos do agente
        self.goals = [
            "Extrair termos semanticamente relevantes",
            "Filtrar ruído e stopwords disfarçadas",
            "Adaptar extração ao contexto"
        ]
    
    @property
    def peas(self) -> PEAS:
        """Especificação PEAS do agente de keywords."""
        return PEAS(
            performance_measures=[
                "Precisão na extração de termos relevantes",
                "Recall de termos discriminativos",
                "Taxa de filtragem de ruído"
            ],
            environment_description=(
                "Matriz TF-IDF do corpus de avaliações. "
                "Vocabulário de termos conhecidos. "
                "Texto original para processamento."
            ),
            actuators=[
                "Extrair lista de keywords ordenadas por relevância",
                "Destacar termos positivos vs negativos",
                "Sinalizar termos fora do vocabulário",
                "Adaptar quantidade de keywords ao contexto"
            ],
            sensors=[
                "Receber texto bruto de avaliação",
                "Calcular scores TF-IDF por termo",
                "Detectar termos ausentes no vocabulário",
                "Observar distribuição de scores"
            ]
        )
    
    def _initialize_beliefs(self) -> None:
        """Inicializa crenças específicas do agente de keywords."""
        super()._initialize_beliefs()
        self.beliefs.update({
            "total_extractions": 0,
            "empty_extractions": 0,
            "average_keywords_per_text": 0.0,
            "oov_alerts": 0
        })
    
    def perceive(self, percept: AgentPercept) -> None:
        """
        Processa percepção e atualiza crenças sobre o texto.
        
        Args:
            percept: Percepção contendo texto a analisar
        """
        text = percept.data.get("text", "")
        top_n = percept.data.get("top_n", 5)
        
        # Limpar texto
        texto_limpo = limpar_texto(text)
        palavras = texto_limpo.split()
        
        # Calcular TF-IDF
        tfidf_matrix = self.vectorizer.transform([texto_limpo])
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Atualizar crenças
        self.beliefs["current_text"] = text
        self.beliefs["current_text_clean"] = texto_limpo
        self.beliefs["text_length"] = len(palavras)
        self.beliefs["tfidf_matrix"] = tfidf_matrix
        self.beliefs["feature_names"] = feature_names
        self.beliefs["requested_top_n"] = top_n
        
        # Verificar cobertura do vocabulário
        vocabulary = set(feature_names)
        words_in_vocab = [w for w in palavras if w in vocabulary]
        oov_words = [w for w in palavras if w not in vocabulary]
        
        self.beliefs["vocabulary_coverage"] = len(words_in_vocab) / max(len(palavras), 1)
        self.beliefs["oov_words"] = oov_words
        self.beliefs["has_content"] = tfidf_matrix.nnz > 0
    
    def can_handle(self, request: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Verifica se pode extrair keywords da requisição.
        
        Args:
            request: Requisição com texto
            
        Returns:
            Tupla (pode_processar, motivo)
        """
        text = request.get("text", "")
        
        if not text or not text.strip():
            return False, "Texto vazio"
        
        return True, "Requisição aceita"
    
    def decide(self) -> Optional[str]:
        """
        Decide qual ação executar baseado nas crenças.
        
        Returns:
            Ação a executar
        """
        # Se não há conteúdo TF-IDF, alertar
        if not self.beliefs.get("has_content", False):
            return "extract_empty"
        
        # Se baixa cobertura de vocabulário, alertar
        if self.beliefs.get("vocabulary_coverage", 0) < 0.3:
            return "extract_with_alert"
        
        # Extração normal
        return "extract"
    
    def act(self, action: str) -> Dict[str, Any]:
        """
        Executa a extração de keywords.
        
        Args:
            action: Tipo de extração a executar
            
        Returns:
            Resultado com keywords extraídas
        """
        self.beliefs["total_extractions"] = self.beliefs.get("total_extractions", 0) + 1
        
        if action == "extract_empty":
            self.beliefs["empty_extractions"] = self.beliefs.get("empty_extractions", 0) + 1
            self.alert(
                issue="Nenhum termo do vocabulário encontrado no texto",
                severity="warning"
            )
            return {
                "success": True,
                "action": action,
                "keywords": [],
                "scores": {},
                "warning": "Texto não contém termos conhecidos do vocabulário"
            }
        
        # Extrair keywords
        tfidf_matrix = self.beliefs.get("tfidf_matrix")
        feature_names = self.beliefs.get("feature_names")
        top_n = self.beliefs.get("requested_top_n", 5)
        
        row = tfidf_matrix[0]
        
        # Obter termos e scores
        tuples = list(zip(row.data, row.indices))
        sorted_tuples = sorted(tuples, key=lambda x: x[0], reverse=True)
        
        # Filtrar por score mínimo
        filtered_tuples = [(score, idx) for score, idx in sorted_tuples 
                          if score >= self.MIN_TFIDF_SCORE]
        
        # Limitar ao top_n
        top_tuples = filtered_tuples[:top_n]
        
        keywords = [feature_names[idx] for score, idx in top_tuples]
        scores = {feature_names[idx]: float(score) for score, idx in top_tuples}
        
        # Atualizar estatísticas
        n = self.beliefs["total_extractions"]
        old_avg = self.beliefs.get("average_keywords_per_text", 0)
        self.beliefs["average_keywords_per_text"] = old_avg + (len(keywords) - old_avg) / n
        
        # Emitir alerta se necessário
        if action == "extract_with_alert":
            self.beliefs["oov_alerts"] = self.beliefs.get("oov_alerts", 0) + 1
            self.alert(
                issue=f"Baixa cobertura de vocabulário: {self.beliefs.get('vocabulary_coverage', 0):.1%}",
                severity="info"
            )
        
        result = {
            "success": True,
            "action": action,
            "keywords": keywords,
            "scores": scores,
            "vocabulary_coverage": self.beliefs.get("vocabulary_coverage", 0),
            "oov_words": self.beliefs.get("oov_words", [])
        }
        
        return result
    
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """
        Interface de alto nível para extração de keywords.
        
        Executa o ciclo completo do agente: perceive → decide → act.
        Mantém compatibilidade com a API anterior.
        
        Args:
            text: Texto da avaliação
            top_n: Quantidade de keywords a retornar
            
        Returns:
            Lista com as top_n palavras mais relevantes
        """
        # Verificar se pode processar
        can_process, reason = self.can_handle({"text": text})
        if not can_process:
            return []
        
        # Executar ciclo do agente
        percept = AgentPercept(
            source="environment",
            data={"text": text, "top_n": top_n}
        )
        
        result = self.run_cycle(percept)
        return result.get("keywords", [])
    
    def extract_keywords_with_scores(self, text: str, top_n: int = 5) -> Dict[str, float]:
        """
        Extrai keywords com seus scores TF-IDF.
        
        Args:
            text: Texto da avaliação
            top_n: Quantidade de keywords a retornar
            
        Returns:
            Dicionário {keyword: score}
        """
        can_process, _ = self.can_handle({"text": text})
        if not can_process:
            return {}
        
        percept = AgentPercept(
            source="environment",
            data={"text": text, "top_n": top_n}
        )
        
        result = self.run_cycle(percept)
        return result.get("scores", {})
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de performance do agente."""
        return {
            "total_extractions": self.beliefs.get("total_extractions", 0),
            "empty_extractions": self.beliefs.get("empty_extractions", 0),
            "oov_alerts": self.beliefs.get("oov_alerts", 0),
            "average_keywords_per_text": self.beliefs.get("average_keywords_per_text", 0.0),
            "empty_rate": (
                self.beliefs.get("empty_extractions", 0) / 
                max(self.beliefs.get("total_extractions", 0), 1)
            )
        }
