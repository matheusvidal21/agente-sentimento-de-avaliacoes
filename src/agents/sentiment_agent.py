"""
Agente de Análise de Sentimento.

Responsável por classificar o sentimento de textos usando modelos
de Machine Learning (Naive Bayes ou Regressão Logística).

Especificação PEAS:
    Performance: Maximizar acurácia e F1-score; Minimizar incerteza
    Environment: Textos de avaliações em português; Vocabulário TF-IDF
    Actuators: Classificar sentimento; Emitir probabilidades; Solicitar validação
    Sensors: Receber texto; Observar confiança; Detectar OOV

Descrição:
    Este agente implementa classificação supervisionada para análise de
    sentimento, utilizando representação TF-IDF do texto. O agente demonstra
    autonomia ao poder recusar classificar textos inadequados e ao solicitar
    validação quando a confiança é baixa.
    
    Referências:
    - Russell, S. & Norvig, P. (2020). Artificial Intelligence: A Modern Approach
"""

from typing import Dict, List, Tuple, Any, Optional
from .base_agent import BaseAgent, PEAS, AgentPercept, Performative
from ..data_preprocessing import limpar_texto


class SentimentAgent(BaseAgent):
    """
    Agente especializado em classificação de sentimento.
    
    Utiliza modelos supervisionados (NB/LR) treinados com TF-IDF
    para inferir se uma avaliação é Positiva, Neutra ou Negativa.
    
    Comportamento autônomo:
    - Recusa textos muito curtos (< 2 palavras)
    - Solicita validação quando confiança < 60%
    - Alerta sobre textos fora do vocabulário
    
    Attributes:
        model: Modelo scikit-learn treinado
        vectorizer: TfidfVectorizer treinado
        labels: Mapeamento de classes
        CONFIDENCE_THRESHOLD: Limiar para solicitar validação
        MIN_TEXT_LENGTH: Tamanho mínimo de texto aceito
    """
    
    # Configurações do agente
    CONFIDENCE_THRESHOLD = 0.60
    MIN_TEXT_LENGTH = 2  # palavras
    
    def __init__(self, model, vectorizer, name: str = "SentimentAgent"):
        """
        Inicializa o agente de sentimento.
        
        Args:
            model: Modelo scikit-learn treinado (MultinomialNB ou LogisticRegression)
            vectorizer: TfidfVectorizer treinado no corpus
            name: Identificador do agente
        """
        super().__init__(name)
        self.model = model
        self.vectorizer = vectorizer
        self.labels = ["Negativo", "Neutro", "Positivo"]
        
        # Objetivos do agente
        self.goals = [
            "Maximizar acurácia de classificação",
            "Minimizar incerteza nas predições",
            "Detectar textos problemáticos"
        ]
    
    @property
    def peas(self) -> PEAS:
        """Especificação PEAS do agente de sentimento."""
        return PEAS(
            performance_measures=[
                "Acurácia de classificação (target: > 90%)",
                "F1-Score ponderado (target: > 90%)",
                "Taxa de predições confiáveis (confiança > 60%)",
                "Taxa de detecção de textos problemáticos"
            ],
            environment_description=(
                "Textos de avaliações de produtos em português brasileiro. "
                "Corpus vetorizado com TF-IDF (unigramas e bigramas). "
                "3 classes: Positivo, Neutro, Negativo."
            ),
            actuators=[
                "Classificar sentimento (emitir label)",
                "Emitir probabilidades por classe",
                "Solicitar validação (quando baixa confiança)",
                "Recusar classificação (texto inadequado)",
                "Gerar explicação de features"
            ],
            sensors=[
                "Receber texto bruto de avaliação",
                "Observar comprimento do texto",
                "Calcular confiança (probabilidade máxima)",
                "Detectar palavras fora do vocabulário (OOV)",
                "Receber feedback do ValidationAgent"
            ]
        )
    
    def _initialize_beliefs(self) -> None:
        """Inicializa crenças específicas do agente de sentimento."""
        super()._initialize_beliefs()
        self.beliefs.update({
            "total_classifications": 0,
            "refused_count": 0,
            "low_confidence_count": 0,
            "oov_detected_count": 0,
            "last_confidence": None,
            "average_confidence": 0.0
        })
    
    def perceive(self, percept: AgentPercept) -> None:
        """
        Processa percepção e atualiza crenças sobre o texto.
        
        Analisa o texto recebido para extrair informações relevantes
        que guiarão a decisão de classificação.
        
        Args:
            percept: Percepção contendo texto e metadados
        """
        text = percept.data.get("text", "")
        
        # Limpar e analisar texto
        texto_limpo = limpar_texto(text)
        palavras = texto_limpo.split()
        
        # Atualizar crenças sobre o texto atual
        self.beliefs["current_text"] = text
        self.beliefs["current_text_clean"] = texto_limpo
        self.beliefs["text_length"] = len(palavras)
        self.beliefs["is_empty"] = len(palavras) == 0
        
        # Detectar OOV (Out of Vocabulary)
        vocabulary = set(self.vectorizer.get_feature_names_out())
        words_in_vocab = [w for w in palavras if w in vocabulary]
        oov_words = [w for w in palavras if w not in vocabulary]
        
        self.beliefs["vocabulary_coverage"] = len(words_in_vocab) / max(len(palavras), 1)
        self.beliefs["oov_words"] = oov_words
        self.beliefs["has_significant_oov"] = self.beliefs["vocabulary_coverage"] < 0.3
    
    def can_handle(self, request: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Verifica autonomamente se pode processar a requisição.
        
        O agente recusa textos que não consegue classificar
        adequadamente, demonstrando autonomia.
        
        Args:
            request: Requisição com texto a classificar
            
        Returns:
            Tupla (pode_processar, motivo)
        """
        text = request.get("text", "")
        texto_limpo = limpar_texto(text)
        palavras = texto_limpo.split()
        
        # Verificar tamanho mínimo
        if len(palavras) < self.MIN_TEXT_LENGTH:
            return False, f"Texto muito curto ({len(palavras)} palavras). Mínimo: {self.MIN_TEXT_LENGTH}"
        
        # Verificar se texto está vazio após limpeza
        if not texto_limpo.strip():
            return False, "Texto vazio após limpeza (sem palavras válidas)"
        
        return True, "Requisição aceita"
    
    def decide(self) -> Optional[str]:
        """
        Decide qual ação executar baseado nas crenças.
        
        Analisa o estado interno para determinar se deve:
        - Recusar a classificação
        - Classificar com alerta
        - Classificar normalmente
        
        Returns:
            Ação a executar
        """
        # Se texto vazio, recusar
        if self.beliefs.get("is_empty", True):
            return "refuse"
        
        # Se muito curto, recusar
        if self.beliefs.get("text_length", 0) < self.MIN_TEXT_LENGTH:
            return "refuse"
        
        # Se muito OOV, alertar mas classificar
        if self.beliefs.get("has_significant_oov", False):
            return "classify_with_alert"
        
        # Classificação normal
        return "classify"
    
    def act(self, action: str) -> Dict[str, Any]:
        """
        Executa a ação decidida.
        
        Args:
            action: Ação a executar (classify, classify_with_alert, refuse)
            
        Returns:
            Resultado da classificação ou recusa
        """
        if action == "refuse":
            self.beliefs["refused_count"] = self.beliefs.get("refused_count", 0) + 1
            reason = "Texto inadequado para classificação"
            if self.beliefs.get("is_empty"):
                reason = "Texto vazio"
            elif self.beliefs.get("text_length", 0) < self.MIN_TEXT_LENGTH:
                reason = f"Texto muito curto ({self.beliefs.get('text_length')} palavras)"
            
            self.refuse(reason)
            return {
                "success": False,
                "action": "refuse",
                "reason": reason
            }
        
        # Executar classificação
        texto_limpo = self.beliefs.get("current_text_clean", "")
        texto_tfidf = self.vectorizer.transform([texto_limpo])
        
        # Predição
        previsao_numerica = self.model.predict(texto_tfidf)[0]
        probabilidades = self.model.predict_proba(texto_tfidf)[0]
        
        label = self.labels[previsao_numerica]
        confianca = float(probabilidades[previsao_numerica])
        
        # Atualizar crenças
        self.beliefs["last_confidence"] = confianca
        self.beliefs["total_classifications"] = self.beliefs.get("total_classifications", 0) + 1
        
        # Atualizar média de confiança
        n = self.beliefs["total_classifications"]
        old_avg = self.beliefs.get("average_confidence", 0)
        self.beliefs["average_confidence"] = old_avg + (confianca - old_avg) / n
        
        # Verificar se precisa solicitar validação
        needs_validation = confianca < self.CONFIDENCE_THRESHOLD
        if needs_validation:
            self.beliefs["low_confidence_count"] = self.beliefs.get("low_confidence_count", 0) + 1
            self.send_message(
                receiver="ValidationAgent",
                performative=Performative.REQUEST,
                content={
                    "request_type": "validate_prediction",
                    "confidence": confianca,
                    "label": label
                }
            )
        
        # Emitir alerta se OOV significativo
        if action == "classify_with_alert":
            self.beliefs["oov_detected_count"] = self.beliefs.get("oov_detected_count", 0) + 1
            self.alert(
                issue=f"Alto OOV detectado: {self.beliefs.get('oov_words', [])}",
                severity="warning"
            )
        
        result = {
            "success": True,
            "action": action,
            "label": label,
            "probabilities": {
                lbl: float(prob) 
                for lbl, prob in zip(self.labels, probabilidades)
            },
            "confidence": confianca,
            "needs_validation": needs_validation,
            "vocabulary_coverage": self.beliefs.get("vocabulary_coverage", 1.0)
        }
        
        return result
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Interface de alto nível para classificação.
        
        Executa o ciclo completo do agente: perceive → decide → act.
        Mantém compatibilidade com a API anterior.
        
        Args:
            text: Texto da avaliação a ser classificada
            
        Returns:
            Dicionário com label e probabilidades por classe
        """
        # Verificar se pode processar
        can_process, reason = self.can_handle({"text": text})
        if not can_process:
            return {
                "label": "Indeterminado",
                "probabilities": {lbl: 0.0 for lbl in self.labels},
                "refused": True,
                "reason": reason
            }
        
        # Executar ciclo do agente
        percept = AgentPercept(
            source="environment",
            data={"text": text}
        )
        result = self.run_cycle(percept)
        
        # Formatar resposta para compatibilidade
        if not result.get("success", False):
            return {
                "label": "Indeterminado",
                "probabilities": {lbl: 0.0 for lbl in self.labels},
                "refused": True,
                "reason": result.get("reason", "Erro desconhecido")
            }
        
        return {
            "label": result["label"],
            "probabilities": result["probabilities"]
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
                explanation.append((word, float(score)))
            
            explanation.sort(key=lambda x: abs(x[1]), reverse=True)
            return explanation

        except Exception as e:
            self.alert(f"Erro ao gerar explicação: {e}", severity="error")
            return []
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas de performance do agente.
        
        Returns:
            Métricas de uso e performance
        """
        total = self.beliefs.get("total_classifications", 0) + self.beliefs.get("refused_count", 0)
        return {
            "total_classifications": self.beliefs.get("total_classifications", 0),
            "refused_count": self.beliefs.get("refused_count", 0),
            "low_confidence_count": self.beliefs.get("low_confidence_count", 0),
            "oov_detected_count": self.beliefs.get("oov_detected_count", 0),
            "average_confidence": self.beliefs.get("average_confidence", 0.0),
            "refusal_rate": self.beliefs.get("refused_count", 0) / max(total, 1)
        }
