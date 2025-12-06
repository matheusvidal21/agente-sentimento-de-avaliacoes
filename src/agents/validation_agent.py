"""
Agente de Validação e Quantificação de Incerteza.

Responsável por avaliar a confiabilidade das predições do SentimentAgent,
detectar casos ambíguos e recomendar intervenção humana quando necessário.

Fundamentação Teórica:
    Este agente implementa conceitos fundamentais de raciocínio probabilístico
    e quantificação da incerteza, alinhados com a ementa do curso de 
    Introdução à Inteligência Artificial. A separação entre classificação
    (SentimentAgent) e validação (ValidationAgent) justifica a arquitetura
    multi-agente, onde agentes especializados colaboram em diferentes
    aspectos do problema.
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from scipy.stats import entropy


class ValidationAgent:
    """
    Agente especializado em quantificação de incerteza e validação de predições.
    
    Implementa técnicas de:
    - Análise de confiança probabilística
    - Cálculo de entropia para detecção de ambiguidade
    - Detecção de anomalias (Out-of-Distribution)
    - Recomendação de revisão humana (Human-in-the-Loop)
    
    Atributos:
        CONFIDENCE_THRESHOLD_HIGH (float): Limiar para alta confiança (75%)
        CONFIDENCE_THRESHOLD_LOW (float): Limiar para baixa confiança (50%)
        ENTROPY_THRESHOLD (float): Limiar de entropia para ambiguidade
        MIN_TEXT_LENGTH (int): Tamanho mínimo de texto esperado
        MAX_TEXT_LENGTH (int): Tamanho máximo de texto esperado
    
    Fundamentação:
        A entropia de Shannon é usada como medida de incerteza:
        H(X) = -Σ p(x) * log₂(p(x))
        
        Quanto maior a entropia, maior a incerteza do modelo sobre a classificação.
        Para 3 classes com distribuição uniforme, H_max = log₂(3) ≈ 1.58 bits.
    """
    
    # Thresholds calibrados para o domínio de análise de sentimentos
    CONFIDENCE_THRESHOLD_HIGH = 0.65   # Alta confiança (reduzido para ser menos rigoroso)
    CONFIDENCE_THRESHOLD_LOW = 0.45    # Baixa confiança
    ENTROPY_THRESHOLD = 1.35           # ~85% da entropia máxima para 3 classes
    MIN_TEXT_LENGTH = 3                # Palavras mínimas
    MAX_TEXT_LENGTH = 500              # Palavras máximas
    
    def __init__(self):
        """
        Inicializa o agente de validação.
        
        Mantém histórico de validações para análise posterior de calibração.
        """
        self.validation_history: List[Dict[str, Any]] = []
        self.labels = ["Negativo", "Neutro", "Positivo"]
    
    def validate(
        self, 
        text: str, 
        sentiment_result: Dict[str, Any],
        model_type: str = "lr"
    ) -> Dict[str, Any]:
        """
        Valida a predição do SentimentAgent e quantifica incerteza.
        
        Realiza análise multi-dimensional da confiabilidade:
        1. Confiança: Probabilidade máxima da classe predita
        2. Entropia: Medida de dispersão das probabilidades
        3. Análise textual: Detecta textos atípicos (OOD)
        4. Spread: Diferença entre maior e menor probabilidade
        
        Args:
            text: Texto original da avaliação
            sentiment_result: Resultado do SentimentAgent contendo:
                - label: Classe predita
                - probabilities: Dict com probabilidades por classe
            model_type: Tipo de modelo usado ("nb" ou "lr")
            
        Returns:
            Dict contendo:
                - status: CONFIAVEL, CONFIANCA_MODERADA, BAIXA_CONFIANCA, AMBIGUO, OOD
                - confianca: Probabilidade máxima (0-1)
                - entropia: Medida de incerteza em bits
                - requer_revisao_humana: Boolean indicando necessidade de intervenção
                - recomendacao: String com ação sugerida
                - detalhes: String formatada com análise completa
                - metricas: Dict com métricas numéricas
        """
        probabilities = sentiment_result["probabilities"]
        predicted_label = sentiment_result["label"]
        
        # 1. Extrair probabilidade máxima (confiança)
        max_prob = max(probabilities.values())
        
        # 2. Calcular entropia de Shannon (incerteza)
        probs_array = np.array(list(probabilities.values()))
        # Adicionar pequeno epsilon para evitar log(0)
        probs_safe = np.clip(probs_array, 1e-10, 1.0)
        entropy_value = entropy(probs_safe, base=2)  # Em bits
        
        # Normalizar entropia (0-1) onde 1 = máxima incerteza
        max_entropy = np.log2(len(probabilities))  # log₂(3) ≈ 1.58 para 3 classes
        normalized_entropy = entropy_value / max_entropy
        
        # 3. Análise de características do texto (OOD simples)
        text_length = len(text.split())
        is_text_anomaly = (
            text_length < self.MIN_TEXT_LENGTH or 
            text_length > self.MAX_TEXT_LENGTH
        )
        
        # 4. Calcular spread de probabilidades (discriminabilidade)
        prob_spread = max(probabilities.values()) - min(probabilities.values())
        
        # 5. Determinar status e recomendação
        status, recommendation, requires_review = self._determine_status(
            max_prob, entropy_value, is_text_anomaly, prob_spread, text_length
        )
        
        # 6. Gerar explicação detalhada
        details = self._generate_details(
            max_prob, entropy_value, normalized_entropy, text_length,
            prob_spread, probabilities, predicted_label, status
        )
        
        # 7. Armazenar no histórico para análise de calibração
        validation_record = {
            "text_length": text_length,
            "confidence": max_prob,
            "entropy": entropy_value,
            "normalized_entropy": normalized_entropy,
            "status": status,
            "predicted_label": predicted_label,
            "model": model_type,
            "requires_review": requires_review
        }
        self.validation_history.append(validation_record)
        
        return {
            "status": status,
            "confianca": max_prob,
            "entropia": entropy_value,
            "entropia_normalizada": normalized_entropy,
            "requer_revisao_humana": requires_review,
            "recomendacao": recommendation,
            "detalhes": details,
            "metricas": {
                "tamanho_texto": text_length,
                "spread_probabilidades": prob_spread,
                "modelo": model_type,
                "entropia_maxima": max_entropy
            }
        }
    
    def _determine_status(
        self, 
        confidence: float, 
        entropy_val: float, 
        is_anomaly: bool,
        prob_spread: float,
        text_length: int
    ) -> Tuple[str, str, bool]:
        """
        Determina o status de validação baseado em múltiplos critérios.
        
        Hierarquia de decisão:
        1. OOD (Out-of-Distribution) - Texto muito curto ou longo
        2. AMBIGUO - Alta entropia indica incerteza do modelo
        3. BAIXA_CONFIANCA - Probabilidade máxima insuficiente
        4. CONFIANCA_MODERADA - Confiança aceitável mas não ideal
        5. CONFIAVEL - Alta confiança, pode prosseguir automaticamente
        
        Args:
            confidence: Probabilidade máxima
            entropy_val: Entropia em bits
            is_anomaly: Se texto é atípico
            prob_spread: Diferença max-min de probabilidades
            text_length: Número de palavras
            
        Returns:
            Tuple (status, recomendacao, requer_revisao)
        """
        # 1. Detecção de Out-of-Distribution
        if is_anomaly:
            if text_length < self.MIN_TEXT_LENGTH:
                return (
                    "OOD",
                    f"Texto muito curto ({text_length} palavras). Insuficiente para análise confiável.",
                    True
                )
            else:
                return (
                    "OOD", 
                    f"Texto muito longo ({text_length} palavras). Pode conter múltiplos sentimentos.",
                    True
                )
        
        # 2. Alta Entropia = Ambiguidade (modelo indeciso)
        if entropy_val > self.ENTROPY_THRESHOLD:
            return (
                "AMBIGUO",
                "Sentimento misto ou indefinido detectado. Probabilidades distribuídas entre classes.",
                True
            )
        
        # 3. Baixa Confiança
        if confidence < self.CONFIDENCE_THRESHOLD_LOW:
            return (
                "BAIXA_CONFIANCA",
                "Confiança insuficiente na predição. Recomenda-se revisão por especialista.",
                True
            )
        
        # 4. Confiança Moderada
        if confidence < self.CONFIDENCE_THRESHOLD_HIGH:
            return (
                "CONFIANCA_MODERADA",
                "Predição aceitável com margem de incerteza. Monitorar resultado.",
                False
            )
        
        # 5. Alta Confiança
        return (
            "CONFIAVEL",
            "Predição altamente confiável. Sistema pode prosseguir automaticamente.",
            False
        )
    
    def _generate_details(
        self, 
        confidence: float, 
        entropy_val: float,
        normalized_entropy: float,
        text_length: int,
        prob_spread: float,
        probabilities: Dict[str, float],
        predicted_label: str,
        status: str
    ) -> str:
        """
        Gera explicação textual detalhada da análise de validação.
        
        Formata as métricas e interpretações de forma legível para
        apresentação na interface e relatórios.
        """
        lines = [
            "📊 **Métricas de Confiabilidade:**",
            f"• Confiança: **{confidence:.1%}** (prob. da classe predita)",
            f"• Entropia: **{entropy_val:.3f}** bits ({normalized_entropy:.1%} da máxima)",
            f"• Spread: **{prob_spread:.3f}** (discriminabilidade)",
            f"• Tamanho: **{text_length}** palavras",
            "",
            "🎯 **Distribuição de Probabilidades:**"
        ]
        
        # Ordenar probabilidades do maior para menor
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        for label, prob in sorted_probs:
            marker = "→" if label == predicted_label else " "
            bar_length = int(prob * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            lines.append(f"{marker} {label}: {bar} {prob:.1%}")
        
        # Interpretação baseada no status
        lines.append("")
        lines.append("💡 **Interpretação:**")
        
        interpretations = {
            "CONFIAVEL": "✅ Modelo altamente confiante. Distribuição de probabilidades bem definida.",
            "CONFIANCA_MODERADA": "⚠️ Confiança aceitável, mas existe incerteza residual entre classes.",
            "BAIXA_CONFIANCA": "❌ Modelo inseguro. Probabilidades próximas entre classes.",
            "AMBIGUO": "🔀 Alta entropia indica possível sentimento misto ou texto ambíguo.",
            "OOD": "⚡ Texto fora do padrão de treinamento. Resultados podem não ser confiáveis."
        }
        lines.append(interpretations.get(status, "Status desconhecido."))
        
        return "\n".join(lines)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retorna estatísticas agregadas do histórico de validações.
        
        Útil para:
        - Análise de calibração do modelo em produção
        - Identificação de padrões de incerteza
        - Monitoramento de taxa de revisão humana
        
        Returns:
            Dict com estatísticas agregadas ou mensagem de erro
        """
        if not self.validation_history:
            return {"message": "Nenhuma validação realizada ainda."}
        
        confidences = [v["confidence"] for v in self.validation_history]
        entropies = [v["entropy"] for v in self.validation_history]
        statuses = [v["status"] for v in self.validation_history]
        
        # Calcular distribuição de status
        status_counts = {}
        for status in set(statuses):
            status_counts[status] = statuses.count(status)
        
        # Taxa de revisão humana
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
            "total_revisoes_necessarias": reviews_needed
        }
    
    def reset_history(self) -> None:
        """Limpa o histórico de validações."""
        self.validation_history = []
