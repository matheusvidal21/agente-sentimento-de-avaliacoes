"""
Agente de Explicabilidade baseado em Explainable AI (XAI).

Responsável por explicar as predições do modelo de machine learning,
identificando quais palavras influenciaram a decisão de classificação
usando os pesos reais aprendidos pelo modelo.

Especificação PEAS:
    Performance: Identificar palavras influentes; Mostrar contribuição real
    Environment: Modelo treinado (NB ou LR); Vetorizador TF-IDF; Texto de entrada
    Actuators: Emitir palavras positivas/negativas; Gerar explicação textual
    Sensors: Receber texto; Acessar coeficientes/log-probs; Acessar vocabulário

Descrição:
    Este agente implementa explicabilidade local de predições usando
    os coeficientes aprendidos pelo modelo. Diferente de TF-IDF puro
    (que mede relevância estatística no corpus), este agente mostra
    a contribuição real de cada feature para a decisão do classificador.
    
    Para Naive Bayes, utiliza as log-probabilidades condicionais
    P(palavra|classe) aprendidas durante o treinamento (feature_log_prob_).
    
    Para Regressão Logística, utiliza os coeficientes que representam
    o peso de cada feature na função de decisão linear (coef_).
    
    A explicabilidade é calculada multiplicando os pesos do modelo
    pelos valores TF-IDF do texto, obtendo a contribuição real de
    cada palavra para a predição final.

Referências:
    - Ribeiro et al. (2016). "Why Should I Trust You?": Explaining
      the Predictions of Any Classifier. KDD.
    - Molnar, C. (2022). Interpretable Machine Learning.
    - Russell, S. & Norvig, P. (2020). AI: A Modern Approach, Cap. 19.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from .base_agent import BaseAgent, PEAS, AgentPercept
from ..data_preprocessing import limpar_texto


class ExplainabilityAgent(BaseAgent):
    """
    Agente especializado em explicabilidade de predições de ML.
    
    Utiliza os coeficientes/log-probabilidades do modelo treinado
    para identificar quais palavras contribuíram para a classificação,
    separando influências positivas e negativas.
    
    Comportamento autônomo:
    - Detecta automaticamente o tipo de modelo (NB vs LR)
    - Calcula contribuição real baseada nos pesos do modelo
    - Adapta explicação ao tipo de classificação
    - Alerta quando texto não contém termos do vocabulário
    
    Attributes:
        model: Modelo treinado (MultinomialNB ou LogisticRegression)
        vectorizer: TfidfVectorizer treinado no corpus
        labels: Lista de classes do modelo
        MIN_CONTRIBUTION: Contribuição mínima para considerar relevante
    """
    
    MIN_CONTRIBUTION = 0.01  # Contribuição mínima para considerar relevante
    
    def __init__(self, model, vectorizer, name: str = "ExplainabilityAgent"):
        """
        Inicializa o agente de explicabilidade.
        
        Args:
            model: Modelo scikit-learn treinado (MultinomialNB ou LogisticRegression)
            vectorizer: TfidfVectorizer treinado no corpus
            name: Identificador do agente
        """
        self.model = model
        self.vectorizer = vectorizer
        self.labels = ["Negativo", "Neutro", "Positivo"]
        
        # Detectar tipo de modelo ANTES de chamar super().__init__()
        # pois _initialize_beliefs é chamado no construtor da classe base
        self._model_type = self._detect_model_type()
        
        super().__init__(name)
        
        # Objetivos do agente
        self.goals = [
            "Identificar palavras que influenciaram a predição",
            "Mostrar contribuição real baseada nos pesos do modelo",
            "Fornecer explicação compreensível ao usuário"
        ]
    
    def _detect_model_type(self) -> str:
        """
        Detecta o tipo de modelo para usar a estratégia correta.
        
        Returns:
            'naive_bayes' ou 'logistic_regression'
        """
        if hasattr(self.model, 'feature_log_prob_'):
            return 'naive_bayes'
        elif hasattr(self.model, 'coef_'):
            return 'logistic_regression'
        else:
            raise ValueError("Modelo não suportado. Use MultinomialNB ou LogisticRegression.")
    
    @property
    def peas(self) -> PEAS:
        """Especificação PEAS do agente de explicabilidade."""
        return PEAS(
            performance_measures=[
                "Identificar corretamente as palavras mais influentes",
                "Mostrar contribuição real baseada nos pesos do modelo",
                "Fornecer explicação compreensível ao usuário"
            ],
            environment_description=(
                "Modelo treinado (NB ou LR). "
                "Vetorizador TF-IDF. "
                "Texto de entrada para explicar."
            ),
            actuators=[
                "Emitir lista de palavras com contribuição positiva",
                "Emitir lista de palavras com contribuição negativa",
                "Gerar explicação textual da predição"
            ],
            sensors=[
                "Receber texto de avaliação",
                "Acessar coeficientes/log-probs do modelo",
                "Acessar vocabulário do vetorizador"
            ]
        )
    
    def _initialize_beliefs(self) -> None:
        """Inicializa crenças específicas do agente de explicabilidade."""
        super()._initialize_beliefs()
        self.beliefs.update({
            "total_explanations": 0,
            "empty_explanations": 0,
            "average_positive_words": 0.0,
            "average_negative_words": 0.0,
            "model_type": self._model_type
        })
    
    def perceive(self, percept: AgentPercept) -> None:
        """
        Processa percepção e atualiza crenças sobre o texto.
        
        Args:
            percept: Percepção contendo texto e classe predita
        """
        text = percept.data.get("text", "")
        predicted_class = percept.data.get("predicted_class", "")
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
        self.beliefs["predicted_class"] = predicted_class
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
        Verifica se pode explicar a predição da requisição.
        
        Args:
            request: Requisição com texto e classe predita
            
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
            return "explain_empty"
        
        # Se baixa cobertura de vocabulário, alertar
        if self.beliefs.get("vocabulary_coverage", 0) < 0.3:
            return "explain_with_alert"
        
        # Explicação normal
        return "explain"
    
    def _get_model_weights(self) -> np.ndarray:
        """
        Obtém os pesos do modelo para calcular contribuições.
        
        Para Naive Bayes: diferença de log-probabilidades entre classes
        Para Logistic Regression: coeficientes diretos
        
        Returns:
            Array de pesos por feature (positivo = contribui para positivo)
        """
        if self._model_type == 'naive_bayes':
            # feature_log_prob_[classe][feature] = log P(feature|classe)
            # Classe 0: Negativo, Classe 1: Neutro, Classe 2: Positivo
            log_probs = self.model.feature_log_prob_
            
            # Calcular diferença: positivo - negativo
            # Valores positivos indicam que a palavra puxa para Positivo
            # Valores negativos indicam que a palavra puxa para Negativo
            weights = log_probs[2] - log_probs[0]  # Positivo - Negativo
            
        elif self._model_type == 'logistic_regression':
            # coef_[classe][feature] = peso da feature para a classe
            # Para classificação multiclasse, cada classe tem seus coeficientes
            coefs = self.model.coef_
            
            # Calcular diferença: positivo - negativo
            weights = coefs[2] - coefs[0]  # Positivo - Negativo
        
        return weights
    
    def _calculate_contributions(self) -> Dict[str, float]:
        """
        Calcula a contribuição de cada palavra do texto para a predição.
        
        A contribuição é calculada como: peso_modelo * valor_tfidf
        
        Returns:
            Dicionário {palavra: contribuição}
        """
        tfidf_matrix = self.beliefs.get("tfidf_matrix")
        feature_names = self.beliefs.get("feature_names")
        
        if tfidf_matrix is None or tfidf_matrix.nnz == 0:
            return {}
        
        # Obter pesos do modelo
        weights = self._get_model_weights()
        
        # Calcular contribuições
        contributions = {}
        row = tfidf_matrix[0]
        
        for tfidf_value, idx in zip(row.data, row.indices):
            word = feature_names[idx]
            # Contribuição = peso do modelo * valor TF-IDF
            contribution = float(weights[idx] * tfidf_value)
            
            if abs(contribution) >= self.MIN_CONTRIBUTION:
                contributions[word] = contribution
        
        return contributions
    
    def act(self, action: str) -> Dict[str, Any]:
        """
        Executa a explicação da predição.
        
        Args:
            action: Tipo de explicação a executar
            
        Returns:
            Resultado com explicação da predição
        """
        self.beliefs["total_explanations"] = self.beliefs.get("total_explanations", 0) + 1
        
        if action == "explain_empty":
            self.beliefs["empty_explanations"] = self.beliefs.get("empty_explanations", 0) + 1
            self.alert(
                issue="Nenhum termo do vocabulário encontrado no texto",
                severity="warning"
            )
            return {
                "success": True,
                "action": action,
                "palavras_positivas": [],
                "palavras_negativas": [],
                "palavra_mais_influente": None,
                "explicacao": "Não foi possível explicar: texto não contém termos conhecidos.",
                "warning": "Texto não contém termos conhecidos do vocabulário"
            }
        
        # Calcular contribuições
        contributions = self._calculate_contributions()
        predicted_class = self.beliefs.get("predicted_class", "")
        top_n = self.beliefs.get("requested_top_n", 5)
        
        # Separar palavras positivas e negativas
        palavras_positivas = [
            (word, score) for word, score in contributions.items() if score > 0
        ]
        palavras_negativas = [
            (word, score) for word, score in contributions.items() if score < 0
        ]
        
        # Ordenar por magnitude
        palavras_positivas.sort(key=lambda x: x[1], reverse=True)
        palavras_negativas.sort(key=lambda x: x[1])  # Mais negativo primeiro
        
        # Limitar ao top_n
        palavras_positivas = palavras_positivas[:top_n]
        palavras_negativas = palavras_negativas[:top_n]
        
        # Encontrar palavra mais influente
        all_words = [(w, abs(s), s) for w, s in contributions.items()]
        if all_words:
            all_words.sort(key=lambda x: x[1], reverse=True)
            palavra_mais_influente = all_words[0][0]
            peso_mais_influente = all_words[0][2]
        else:
            palavra_mais_influente = None
            peso_mais_influente = 0
        
        # Gerar explicação textual
        explicacao = self._generate_explanation(
            predicted_class, 
            palavra_mais_influente, 
            peso_mais_influente,
            palavras_positivas,
            palavras_negativas
        )
        
        # Atualizar estatísticas
        n = self.beliefs["total_explanations"]
        old_avg_pos = self.beliefs.get("average_positive_words", 0)
        old_avg_neg = self.beliefs.get("average_negative_words", 0)
        self.beliefs["average_positive_words"] = old_avg_pos + (len(palavras_positivas) - old_avg_pos) / n
        self.beliefs["average_negative_words"] = old_avg_neg + (len(palavras_negativas) - old_avg_neg) / n
        
        # Emitir alerta se necessário
        if action == "explain_with_alert":
            self.alert(
                issue=f"Baixa cobertura de vocabulário: {self.beliefs.get('vocabulary_coverage', 0):.1%}",
                severity="info"
            )
        
        result = {
            "success": True,
            "action": action,
            "palavras_positivas": palavras_positivas,
            "palavras_negativas": palavras_negativas,
            "palavra_mais_influente": palavra_mais_influente,
            "explicacao": explicacao,
            "vocabulary_coverage": self.beliefs.get("vocabulary_coverage", 0),
            "oov_words": self.beliefs.get("oov_words", []),
            "model_type": self._model_type
        }
        
        return result
    
    def _generate_explanation(
        self, 
        predicted_class: str, 
        palavra_mais_influente: Optional[str],
        peso: float,
        palavras_positivas: List[Tuple[str, float]],
        palavras_negativas: List[Tuple[str, float]]
    ) -> str:
        """
        Gera uma explicação textual da predição.
        
        Args:
            predicted_class: Classe predita pelo modelo
            palavra_mais_influente: Palavra com maior contribuição absoluta
            peso: Peso da palavra mais influente
            palavras_positivas: Lista de (palavra, contribuição positiva)
            palavras_negativas: Lista de (palavra, contribuição negativa)
            
        Returns:
            Explicação em linguagem natural
        """
        if not palavra_mais_influente:
            return "Não foi possível identificar palavras influentes na predição."
        
        # Construir explicação
        explicacao = f"O modelo classificou como {predicted_class} porque "
        
        if predicted_class == "Positivo":
            if palavras_positivas:
                top_positivas = [w for w, _ in palavras_positivas[:3]]
                explicacao += f"as palavras {', '.join(top_positivas)} tiveram contribuição positiva."
            else:
                explicacao += "não encontrou palavras negativas suficientes para mudar a classificação."
                
        elif predicted_class == "Negativo":
            if palavras_negativas:
                top_negativas = [w for w, _ in palavras_negativas[:3]]
                explicacao += f"as palavras {', '.join(top_negativas)} tiveram contribuição negativa."
            else:
                explicacao += "não encontrou palavras positivas suficientes para mudar a classificação."
                
        else:  # Neutro
            explicacao += "as contribuições positivas e negativas se equilibraram."
            if palavras_positivas and palavras_negativas:
                explicacao += f" Palavras como '{palavras_positivas[0][0]}' (+) e '{palavras_negativas[0][0]}' (-) se compensaram."
        
        # Adicionar destaque para palavra mais influente
        explicacao += f" A palavra mais influente foi '{palavra_mais_influente}' (peso: {peso:.2f})."
        
        return explicacao
    
    def explain_prediction(self, text: str, predicted_class: str, top_n: int = 5) -> Dict[str, Any]:
        """
        Explica por que o modelo classificou o texto daquela forma.
        
        Interface de alto nível para explicação de predições.
        Executa o ciclo completo do agente: perceive → decide → act.
        
        Args:
            text: Texto da avaliação
            predicted_class: Classe predita (Positivo/Negativo/Neutro)
            top_n: Quantidade de palavras a retornar por categoria
            
        Returns:
            Dicionário com palavras positivas, negativas e explicação:
            {
                "palavras_positivas": [("excelente", 0.85), ...],
                "palavras_negativas": [("péssimo", -0.91), ...],
                "palavra_mais_influente": "péssimo",
                "explicacao": "O modelo classificou como..."
            }
        """
        # Verificar se pode processar
        can_process, reason = self.can_handle({"text": text})
        if not can_process:
            return {
                "palavras_positivas": [],
                "palavras_negativas": [],
                "palavra_mais_influente": None,
                "explicacao": f"Não foi possível explicar: {reason}"
            }
        
        # Executar ciclo do agente
        percept = AgentPercept(
            source="environment",
            data={
                "text": text,
                "predicted_class": predicted_class,
                "top_n": top_n
            }
        )
        
        result = self.run_cycle(percept)
        
        return {
            "palavras_positivas": result.get("palavras_positivas", []),
            "palavras_negativas": result.get("palavras_negativas", []),
            "palavra_mais_influente": result.get("palavra_mais_influente"),
            "explicacao": result.get("explicacao", ""),
            "model_type": result.get("model_type", self._model_type)
        }
    
    def get_word_contributions(self, text: str) -> Dict[str, float]:
        """
        Obtém contribuições de todas as palavras do texto.
        
        Args:
            text: Texto da avaliação
            
        Returns:
            Dicionário {palavra: contribuição}
        """
        can_process, _ = self.can_handle({"text": text})
        if not can_process:
            return {}
        
        percept = AgentPercept(
            source="environment",
            data={"text": text, "predicted_class": "", "top_n": 100}
        )
        
        self.perceive(percept)
        return self._calculate_contributions()
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de performance do agente."""
        return {
            "total_explanations": self.beliefs.get("total_explanations", 0),
            "empty_explanations": self.beliefs.get("empty_explanations", 0),
            "average_positive_words": self.beliefs.get("average_positive_words", 0.0),
            "average_negative_words": self.beliefs.get("average_negative_words", 0.0),
            "model_type": self._model_type,
            "empty_rate": (
                self.beliefs.get("empty_explanations", 0) / 
                max(self.beliefs.get("total_explanations", 0), 1)
            )
        }
