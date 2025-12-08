"""
Agente de Validação e Quantificação de Incerteza.

Responsável por avaliar a confiabilidade das predições do SentimentAgent,
detectar casos ambíguos e recomendar intervenção humana quando necessário.

Especificação PEAS:
    Performance: Minimizar falsos positivos de alta confiança; Detectar 100% OOD
    Environment: Distribuições de probabilidade; Histórico de validações
    Actuators: Emitir status; Recomendar revisão; Ajustar thresholds
    Sensors: Receber probabilidades; Calcular entropia; Detectar anomalias

Descrição:
    Este agente implementa conceitos fundamentais de raciocínio probabilístico
    e quantificação da incerteza. A entropia de Shannon é usada como medida
    de incerteza: H(X) = -Σ p(x) * log₂(p(x))
    
    Referências:
    - Russell, S. & Norvig, P. (2020). Artificial Intelligence: A Modern Approach
    - Shannon, C. (1948). A Mathematical Theory of Communication
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from scipy.stats import entropy
from .base_agent import BaseAgent, PEAS, AgentPercept, Performative


class ValidationAgent(BaseAgent):
    """
    Agente especializado em quantificação de incerteza e validação de predições.
    
    Implementa técnicas de:
    - Análise de confiança probabilística
    - Cálculo de entropia para detecção de ambiguidade
    - Detecção de anomalias (Out-of-Distribution)
    - Recomendação de revisão humana (Human-in-the-Loop)
    
    Comportamento autônomo:
    - Auto-calibra thresholds com base no histórico
    - Alerta proativamente quando detecta padrões anômalos
    - Pode solicitar re-análise ao SentimentAgent
    
    Attributes:
        CONFIDENCE_THRESHOLD_HIGH: Limiar para alta confiança (65%)
        CONFIDENCE_THRESHOLD_LOW: Limiar para baixa confiança (45%)
        ENTROPY_THRESHOLD: Limiar de entropia para ambiguidade
        MIN_TEXT_LENGTH: Tamanho mínimo de texto esperado
        MAX_TEXT_LENGTH: Tamanho máximo de texto esperado
    """
    
    # Thresholds calibrados para o domínio de análise de sentimentos
    CONFIDENCE_THRESHOLD_HIGH = 0.65
    CONFIDENCE_THRESHOLD_LOW = 0.45
    ENTROPY_THRESHOLD = 1.35  # ~85% da entropia máxima para 3 classes
    MIN_TEXT_LENGTH = 3       # Palavras mínimas
    MAX_TEXT_LENGTH = 500     # Palavras máximas
    
    # Thresholds para auto-calibração
    CALIBRATION_WINDOW = 50   # Janela de histórico para calibração
    
    def __init__(self, name: str = "ValidationAgent"):
        """
        Inicializa o agente de validação.
        
        Args:
            name: Identificador do agente
        """
        super().__init__(name)
        self.validation_history: List[Dict[str, Any]] = []
        self.labels = ["Negativo", "Neutro", "Positivo"]
        
        # Objetivos do agente
        self.goals = [
            "Minimizar falsos positivos em predições de alta confiança",
            "Detectar 100% dos casos Out-of-Distribution",
            "Manter taxa de revisão humana < 20%",
            "Auto-calibrar thresholds para o domínio"
        ]
    
    @property
    def peas(self) -> PEAS:
        """Especificação PEAS do agente de validação."""
        return PEAS(
            performance_measures=[
                "Taxa de detecção de OOD (target: 100%)",
                "Precisão na classificação de confiança",
                "Taxa de revisão humana (target: < 20%)",
                "Calibração de probabilidades (ECE < 0.05)"
            ],
            environment_description=(
                "Distribuições de probabilidade do SentimentAgent. "
                "Histórico de validações anteriores. "
                "Thresholds calibrados para o domínio."
            ),
            actuators=[
                "Emitir status de confiança (5 níveis)",
                "Recomendar revisão humana",
                "Ajustar thresholds dinamicamente",
                "Solicitar re-análise ao SentimentAgent",
                "Alertar sobre padrões anômalos"
            ],
            sensors=[
                "Receber probabilidades por classe",
                "Calcular entropia de Shannon",
                "Observar comprimento do texto",
                "Monitorar histórico de validações",
                "Detectar drift de distribuição"
            ]
        )
    
    def _initialize_beliefs(self) -> None:
        """Inicializa crenças específicas do agente de validação."""
        super()._initialize_beliefs()
        self.beliefs.update({
            "total_validations": 0,
            "reviews_requested": 0,
            "ood_detected": 0,
            "ambiguous_detected": 0,
            "current_confidence_threshold_high": self.CONFIDENCE_THRESHOLD_HIGH,
            "current_confidence_threshold_low": self.CONFIDENCE_THRESHOLD_LOW,
            "average_entropy": 0.0,
            "consecutive_low_confidence": 0
        })
    
    def perceive(self, percept: AgentPercept) -> None:
        """
        Processa percepção e atualiza crenças sobre a predição.
        
        Analisa as probabilidades e metadados recebidos para
        formar crenças sobre a qualidade da predição.
        
        Args:
            percept: Percepção contendo probabilidades e metadados
        """
        data = percept.data
        
        # Extrair probabilidades
        probabilities = data.get("probabilities", {})
        predicted_label = data.get("label", "")
        text = data.get("text", "")
        model_type = data.get("model_type", "lr")
        
        # Calcular métricas
        probs_array = np.array(list(probabilities.values()))
        probs_safe = np.clip(probs_array, 1e-10, 1.0)
        
        max_prob = float(max(probabilities.values()))
        entropy_value = float(entropy(probs_safe, base=2))
        max_entropy = np.log2(len(probabilities))
        normalized_entropy = entropy_value / max_entropy
        prob_spread = max(probabilities.values()) - min(probabilities.values())
        
        # Análise do texto
        text_length = len(text.split())
        is_text_anomaly = (
            text_length < self.MIN_TEXT_LENGTH or 
            text_length > self.MAX_TEXT_LENGTH
        )
        
        # Atualizar crenças
        self.beliefs["current_probabilities"] = probabilities
        self.beliefs["current_label"] = predicted_label
        self.beliefs["current_confidence"] = max_prob
        self.beliefs["current_entropy"] = entropy_value
        self.beliefs["normalized_entropy"] = normalized_entropy
        self.beliefs["prob_spread"] = prob_spread
        self.beliefs["text_length"] = text_length
        self.beliefs["is_text_anomaly"] = is_text_anomaly
        self.beliefs["model_type"] = model_type
        self.beliefs["text"] = text
    
    def can_handle(self, request: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Verifica se pode validar a requisição.
        
        Args:
            request: Requisição de validação
            
        Returns:
            Tupla (pode_processar, motivo)
        """
        probabilities = request.get("probabilities", {})
        
        if not probabilities:
            return False, "Probabilidades não fornecidas"
        
        if len(probabilities) != 3:
            return False, f"Esperado 3 classes, recebido {len(probabilities)}"
        
        return True, "Requisição aceita"
    
    def decide(self) -> Optional[str]:
        """
        Decide qual ação executar baseado nas crenças.
        
        Hierarquia de decisão:
        1. OOD (Out-of-Distribution) - Texto atípico
        2. AMBIGUO - Alta entropia
        3. BAIXA_CONFIANCA - Probabilidade insuficiente
        4. CONFIANCA_MODERADA - Aceitável com ressalvas
        5. CONFIAVEL - Alta confiança
        
        Returns:
            Ação a executar
        """
        # Verificar anomalia de texto (OOD)
        if self.beliefs.get("is_text_anomaly", False):
            return "validate_ood"
        
        # Verificar entropia alta (ambiguidade)
        if self.beliefs.get("current_entropy", 0) > self.ENTROPY_THRESHOLD:
            return "validate_ambiguous"
        
        # Verificar confiança
        confidence = self.beliefs.get("current_confidence", 0)
        threshold_low = self.beliefs.get("current_confidence_threshold_low", self.CONFIDENCE_THRESHOLD_LOW)
        threshold_high = self.beliefs.get("current_confidence_threshold_high", self.CONFIDENCE_THRESHOLD_HIGH)
        
        if confidence < threshold_low:
            return "validate_low_confidence"
        elif confidence < threshold_high:
            return "validate_moderate_confidence"
        else:
            return "validate_high_confidence"
    
    def act(self, action: str) -> Dict[str, Any]:
        """
        Executa a validação e gera resultado.
        
        Args:
            action: Tipo de validação a executar
            
        Returns:
            Resultado da validação com status e recomendações
        """
        status_map = {
            "validate_ood": "OOD",
            "validate_ambiguous": "AMBIGUO",
            "validate_low_confidence": "BAIXA_CONFIANCA",
            "validate_moderate_confidence": "CONFIANCA_MODERADA",
            "validate_high_confidence": "CONFIAVEL"
        }
        
        status = status_map.get(action, "DESCONHECIDO")
        requires_review = action in ["validate_ood", "validate_ambiguous", "validate_low_confidence"]
        
        # Gerar recomendação
        recommendation = self._generate_recommendation(action)
        
        # Atualizar estatísticas
        self.beliefs["total_validations"] = self.beliefs.get("total_validations", 0) + 1
        
        if requires_review:
            self.beliefs["reviews_requested"] = self.beliefs.get("reviews_requested", 0) + 1
            
            # Rastrear baixa confiança consecutiva
            self.beliefs["consecutive_low_confidence"] = self.beliefs.get("consecutive_low_confidence", 0) + 1
            
            # Alerta proativo se muitas baixa confiança consecutivas
            if self.beliefs["consecutive_low_confidence"] >= 3:
                self.alert(
                    issue=f"{self.beliefs['consecutive_low_confidence']} predições consecutivas de baixa confiança",
                    severity="warning"
                )
        else:
            self.beliefs["consecutive_low_confidence"] = 0
        
        if action == "validate_ood":
            self.beliefs["ood_detected"] = self.beliefs.get("ood_detected", 0) + 1
        elif action == "validate_ambiguous":
            self.beliefs["ambiguous_detected"] = self.beliefs.get("ambiguous_detected", 0) + 1
        
        # Atualizar média de entropia
        n = self.beliefs["total_validations"]
        old_avg = self.beliefs.get("average_entropy", 0)
        current_entropy = self.beliefs.get("current_entropy", 0)
        self.beliefs["average_entropy"] = old_avg + (current_entropy - old_avg) / n
        
        # Armazenar no histórico
        validation_record = {
            "text_length": self.beliefs.get("text_length", 0),
            "confidence": self.beliefs.get("current_confidence", 0),
            "entropy": current_entropy,
            "normalized_entropy": self.beliefs.get("normalized_entropy", 0),
            "status": status,
            "predicted_label": self.beliefs.get("current_label", ""),
            "model": self.beliefs.get("model_type", ""),
            "requires_review": requires_review
        }
        self.validation_history.append(validation_record)
        
        # Auto-calibração periódica
        if len(self.validation_history) % self.CALIBRATION_WINDOW == 0:
            self._auto_calibrate()
        
        # Gerar detalhes formatados
        details = self._generate_details(status)
        
        result = {
            "success": True,
            "status": status,
            "confianca": self.beliefs.get("current_confidence", 0),
            "entropia": current_entropy,
            "entropia_normalizada": self.beliefs.get("normalized_entropy", 0),
            "requer_revisao_humana": requires_review,
            "recomendacao": recommendation,
            "detalhes": details,
            "metricas": {
                "tamanho_texto": self.beliefs.get("text_length", 0),
                "spread_probabilidades": self.beliefs.get("prob_spread", 0),
                "modelo": self.beliefs.get("model_type", ""),
                "entropia_maxima": np.log2(3)
            }
        }
        
        # Notificar outros agentes se necessário
        if requires_review:
            self.send_message(
                receiver="ActionAgent",
                performative=Performative.INFORM,
                content={
                    "validation_status": status,
                    "requires_human_review": True,
                    "confidence": self.beliefs.get("current_confidence", 0)
                }
            )
        
        return result
    
    def _generate_recommendation(self, action: str) -> str:
        """Gera recomendação textual baseada na ação."""
        text_length = self.beliefs.get("text_length", 0)
        
        recommendations = {
            "validate_ood": (
                f"Texto fora do padrão ({text_length} palavras). "
                f"{'Insuficiente para análise.' if text_length < self.MIN_TEXT_LENGTH else 'Pode conter múltiplos sentimentos.'}"
            ),
            "validate_ambiguous": (
                "Sentimento misto ou indefinido detectado. "
                "Probabilidades distribuídas entre classes."
            ),
            "validate_low_confidence": (
                "Confiança insuficiente na predição. "
                "Recomenda-se revisão por especialista."
            ),
            "validate_moderate_confidence": (
                "Predição aceitável com margem de incerteza. "
                "Monitorar resultado."
            ),
            "validate_high_confidence": (
                "Predição altamente confiável. "
                "Sistema pode prosseguir automaticamente."
            )
        }
        
        return recommendations.get(action, "Status desconhecido.")
    
    def _generate_details(self, status: str) -> str:
        """Gera explicação textual detalhada da validação."""
        probabilities = self.beliefs.get("current_probabilities", {})
        predicted_label = self.beliefs.get("current_label", "")
        confidence = self.beliefs.get("current_confidence", 0)
        entropy_val = self.beliefs.get("current_entropy", 0)
        normalized_entropy = self.beliefs.get("normalized_entropy", 0)
        prob_spread = self.beliefs.get("prob_spread", 0)
        text_length = self.beliefs.get("text_length", 0)
        
        lines = [
            "📊 **Métricas de Confiabilidade:**",
            f"• Confiança: **{confidence:.1%}** (prob. da classe predita)",
            f"• Entropia: **{entropy_val:.3f}** bits ({normalized_entropy:.1%} da máxima)",
            f"• Spread: **{prob_spread:.3f}** (discriminabilidade)",
            f"• Tamanho: **{text_length}** palavras",
            "",
            "🎯 **Distribuição de Probabilidades:**"
        ]
        
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        for label, prob in sorted_probs:
            marker = "→" if label == predicted_label else " "
            bar_length = int(prob * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            lines.append(f"{marker} {label}: {bar} {prob:.1%}")
        
        lines.append("")
        lines.append("💡 **Interpretação:**")
        
        interpretations = {
            "CONFIAVEL": "Modelo altamente confiante. Distribuição de probabilidades bem definida.",
            "CONFIANCA_MODERADA": "Confiança aceitável, mas existe incerteza residual entre classes.",
            "BAIXA_CONFIANCA": "Modelo inseguro. Probabilidades próximas entre classes.",
            "AMBIGUO": "Alta entropia indica possível sentimento misto ou texto ambíguo.",
            "OOD": "⚡ Texto fora do padrão de treinamento. Resultados podem não ser confiáveis."
        }
        lines.append(interpretations.get(status, "Status desconhecido."))
        
        return "\n".join(lines)
    
    def _auto_calibrate(self) -> None:
        """
        Auto-calibra thresholds baseado no histórico recente.
        
        Implementa proatividade: o agente ajusta seus próprios
        parâmetros para melhorar performance.
        """
        if len(self.validation_history) < self.CALIBRATION_WINDOW:
            return
        
        recent = self.validation_history[-self.CALIBRATION_WINDOW:]
        review_rate = sum(1 for v in recent if v["requires_review"]) / len(recent)
        
        # Se taxa de revisão muito alta, relaxar thresholds
        if review_rate > 0.3:
            self.beliefs["current_confidence_threshold_high"] = max(
                0.55, self.beliefs["current_confidence_threshold_high"] - 0.05
            )
            self.beliefs["current_confidence_threshold_low"] = max(
                0.35, self.beliefs["current_confidence_threshold_low"] - 0.05
            )
            self.alert(
                issue=f"Taxa de revisão alta ({review_rate:.1%}). Thresholds relaxados.",
                severity="info"
            )
        
        # Se taxa muito baixa, pode haver risco de falsos positivos
        elif review_rate < 0.05:
            self.beliefs["current_confidence_threshold_high"] = min(
                0.80, self.beliefs["current_confidence_threshold_high"] + 0.03
            )
            self.alert(
                issue=f"Taxa de revisão muito baixa ({review_rate:.1%}). Thresholds ajustados.",
                severity="info"
            )
    
    def validate(
        self, 
        text: str, 
        sentiment_result: Dict[str, Any],
        model_type: str = "lr"
    ) -> Dict[str, Any]:
        """
        Interface de alto nível para validação.
        
        Executa o ciclo completo do agente: perceive → decide → act.
        Mantém compatibilidade com a API anterior.
        
        Args:
            text: Texto original da avaliação
            sentiment_result: Resultado do SentimentAgent
            model_type: Tipo de modelo usado
            
        Returns:
            Resultado da validação
        """
        # Verificar se pode processar
        can_process, reason = self.can_handle(sentiment_result)
        if not can_process:
            return {
                "status": "ERRO",
                "confianca": 0,
                "entropia": 0,
                "entropia_normalizada": 0,
                "requer_revisao_humana": True,
                "recomendacao": reason,
                "detalhes": f"Erro: {reason}",
                "metricas": {}
            }
        
        # Executar ciclo do agente
        percept = AgentPercept(
            source="SentimentAgent",
            data={
                "text": text,
                "probabilities": sentiment_result.get("probabilities", {}),
                "label": sentiment_result.get("label", ""),
                "model_type": model_type
            }
        )
        
        result = self.run_cycle(percept)
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retorna estatísticas agregadas do histórico de validações.
        
        Returns:
            Dict com estatísticas de performance do agente
        """
        if not self.validation_history:
            return {"message": "Nenhuma validação realizada ainda."}
        
        confidences = [v["confidence"] for v in self.validation_history]
        entropies = [v["entropy"] for v in self.validation_history]
        statuses = [v["status"] for v in self.validation_history]
        
        status_counts = {}
        for status in set(statuses):
            status_counts[status] = statuses.count(status)
        
        reviews_needed = sum(1 for v in self.validation_history if v["requires_review"])
        review_rate = reviews_needed / len(self.validation_history)
        
        return {
            "total_validacoes": len(self.validation_history),
            "confianca": {
                "media": float(np.mean(confidences)),
                "std": float(np.std(confidences)),
                "min": float(np.min(confidences)),
                "max": float(np.max(confidences))
            },
            "entropia": {
                "media": float(np.mean(entropies)),
                "std": float(np.std(entropies))
            },
            "distribuicao_status": status_counts,
            "taxa_revisao_humana": review_rate,
            "total_revisoes_necessarias": reviews_needed,
            "thresholds_atuais": {
                "high": self.beliefs.get("current_confidence_threshold_high", self.CONFIDENCE_THRESHOLD_HIGH),
                "low": self.beliefs.get("current_confidence_threshold_low", self.CONFIDENCE_THRESHOLD_LOW)
            }
        }
    
    def reset_history(self) -> None:
        """Limpa o histórico de validações."""
        self.validation_history = []
        self._initialize_beliefs()
    
    def compare_predictions(
        self,
        text: str,
        nb_result: Dict[str, Any],
        lr_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compara predições de dois modelos e escolhe a mais confiável.
        
        O agente validador pondera entre os resultados dos modelos
        Naive Bayes e Regressão Logística baseado em:
        1. Confiança (probabilidade máxima)
        2. Entropia (menor entropia = mais certeza)
        3. Spread de probabilidades (maior separação = melhor discriminação)
        
        Args:
            text: Texto original da avaliação
            nb_result: Resultado do SentimentAgent com Naive Bayes
            lr_result: Resultado do SentimentAgent com Regressão Logística
            
        Returns:
            Dict com o resultado escolhido e justificativa da decisão
        """
        # Calcular métricas para Naive Bayes
        nb_probs = np.array(list(nb_result['probabilities'].values()))
        nb_probs_safe = np.clip(nb_probs, 1e-10, 1.0)
        nb_confidence = float(max(nb_result['probabilities'].values()))
        nb_entropy = float(entropy(nb_probs_safe, base=2))
        nb_spread = float(max(nb_result['probabilities'].values()) - min(nb_result['probabilities'].values()))
        
        # Calcular métricas para Regressão Logística
        lr_probs = np.array(list(lr_result['probabilities'].values()))
        lr_probs_safe = np.clip(lr_probs, 1e-10, 1.0)
        lr_confidence = float(max(lr_result['probabilities'].values()))
        lr_entropy = float(entropy(lr_probs_safe, base=2))
        lr_spread = float(max(lr_result['probabilities'].values()) - min(lr_result['probabilities'].values()))
        
        # Sistema de pontuação ponderada
        # Pesos: Confiança (40%), Entropia inversa (35%), Spread (25%)
        max_entropy = np.log2(3)  # Entropia máxima para 3 classes
        
        # Score NB: maior confiança + menor entropia + maior spread
        nb_score = (
            0.40 * nb_confidence +
            0.35 * (1 - nb_entropy / max_entropy) +
            0.25 * nb_spread
        )
        
        # Score LR
        lr_score = (
            0.40 * lr_confidence +
            0.35 * (1 - lr_entropy / max_entropy) +
            0.25 * lr_spread
        )
        
        # Decisão: escolher o modelo com maior score
        if nb_score > lr_score:
            chosen_model = "nb"
            chosen_result = nb_result
            chosen_confidence = nb_confidence
            chosen_entropy = nb_entropy
            rejected_model = "lr"
            rejected_score = lr_score
            chosen_score = nb_score
        else:
            chosen_model = "lr"
            chosen_result = lr_result
            chosen_confidence = lr_confidence
            chosen_entropy = lr_entropy
            rejected_model = "nb"
            rejected_score = nb_score
            chosen_score = lr_score
        
        # Verificar se os modelos concordam
        models_agree = nb_result['label'] == lr_result['label']
        
        # Calcular diferença de score (margem de decisão)
        score_diff = abs(nb_score - lr_score)
        decision_confidence = "alta" if score_diff > 0.1 else "moderada" if score_diff > 0.05 else "baixa"
        
        # Gerar justificativa
        justificativa_parts = []
        
        if models_agree:
            justificativa_parts.append(
                f"Ambos os modelos concordam: **{chosen_result['label']}**. "
                f"Escolhido {chosen_model.upper()} por ter métricas superiores."
            )
        else:
            justificativa_parts.append(
                f"Modelos divergem: NB={nb_result['label']}, LR={lr_result['label']}. "
                f"Escolhido **{chosen_model.upper()}** ({chosen_result['label']}) por pontuação superior."
            )
        
        justificativa_parts.append(
            f"\n\nComparacao de Scores:\n"
            f"- Naive Bayes: {nb_score:.3f} (conf: {nb_confidence:.1%}, ent: {nb_entropy:.3f})\n"
            f"- Reg. Logistica: {lr_score:.3f} (conf: {lr_confidence:.1%}, ent: {lr_entropy:.3f})\n"
            f"- Margem de decisao: {score_diff:.3f} ({decision_confidence})"
        )
        
        if not models_agree and decision_confidence == "baixa":
            justificativa_parts.append(
                "\n\nAtencao: Modelos divergentes com margem baixa. "
                "Recomenda-se revisao humana."
            )
        
        return {
            "chosen_model": chosen_model,
            "chosen_result": chosen_result,
            "nb_result": nb_result,
            "lr_result": lr_result,
            "models_agree": models_agree,
            "comparison": {
                "nb": {
                    "score": nb_score,
                    "confidence": nb_confidence,
                    "entropy": nb_entropy,
                    "spread": nb_spread,
                    "label": nb_result['label']
                },
                "lr": {
                    "score": lr_score,
                    "confidence": lr_confidence,
                    "entropy": lr_entropy,
                    "spread": lr_spread,
                    "label": lr_result['label']
                }
            },
            "decision_confidence": decision_confidence,
            "score_difference": score_diff,
            "justificativa": "\n".join(justificativa_parts),
            "requires_human_review": not models_agree and decision_confidence == "baixa"
        }
