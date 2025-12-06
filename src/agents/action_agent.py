"""
Agente de Ação.

Responsável por definir ações táticas baseadas em regras de negócio,
considerando o sentimento e o status de validação da predição.

Fundamentação:
    O ActionAgent implementa um sistema de tomada de decisão que considera
    não apenas o sentimento detectado, mas também a confiabilidade dessa
    predição (fornecida pelo ValidationAgent). Isso demonstra coordenação
    real entre agentes, onde a decisão de um agente depende da saída de outro.
"""

from typing import Set


class ActionAgent:
    """
    Agente especializado em recomendação de ações táticas.
    
    Implementa regras de negócio para determinar qual ação tomar
    com base no sentimento identificado e no status de validação.
    
    A lógica de decisão considera:
    1. Status de validação (confiabilidade da predição)
    2. Sentimento detectado (Positivo/Neutro/Negativo)
    
    Casos de baixa confiança são sempre escalados para revisão humana,
    implementando um padrão Human-in-the-Loop.
    """
    
    # Status que requerem intervenção humana
    HUMAN_REVIEW_STATUSES: Set[str] = {"BAIXA_CONFIANCA", "AMBIGUO", "OOD"}

    def get_action(self, sentiment: str, validation_status: str) -> str:
        """
        Define a ação apropriada com base no sentimento e status de validação.
        
        A decisão segue uma hierarquia:
        1. Se validação indica baixa confiança → escalar para humano
        2. Se confiável e positivo → ação automática de agradecimento
        3. Se confiável e negativo → ação prioritária de atendimento
        4. Outros casos → ação moderada com monitoramento
        
        Args:
            sentiment: Sentimento identificado pelo SentimentAgent
            validation_status: Status de confiabilidade do ValidationAgent
                Valores possíveis: CONFIAVEL, CONFIANCA_MODERADA, 
                BAIXA_CONFIANCA, AMBIGUO, OOD
            
        Returns:
            String com a ação recomendada
        """
        # Casos de baixa confiança: sempre escalar para humano
        if validation_status in self.HUMAN_REVIEW_STATUSES:
            return "⚠️ Encaminhar para revisão humana - predição requer validação manual."
        
        # Sentimento positivo com alta confiança
        if sentiment == "Positivo" and validation_status == "CONFIAVEL":
            return "✅ Agradecer automaticamente e incentivar novas compras."
        
        # Sentimento positivo com confiança moderada
        if sentiment == "Positivo":
            return "👍 Agradecer com supervisão posterior."
        
        # Sentimento negativo com alta confiança
        if sentiment == "Negativo" and validation_status == "CONFIAVEL":
            return "🔴 Priorizar atendimento - cliente insatisfeito confirmado."
        
        # Sentimento negativo com confiança moderada
        if sentiment == "Negativo":
            return "⚠️ Atender com cautela - possível insatisfação detectada."
        
        # Neutro ou outros casos
        if validation_status == "CONFIAVEL":
            return "📊 Registrar feedback neutro e monitorar padrões."
        
        return "📝 Registrar para análise posterior."
