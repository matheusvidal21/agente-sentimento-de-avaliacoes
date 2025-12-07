"""
Agente de Ação.

Responsável por definir ações táticas baseadas em regras de negócio,
considerando o sentimento e o status de validação da predição.

Especificação PEAS:
    Performance: Maximizar satisfação do cliente; Otimizar alocação de recursos
    Environment: Sentimento classificado; Status de validação; Regras de negócio
    Actuators: Emitir ação recomendada; Escalar para humano; Priorizar atendimento
    Sensors: Receber sentimento; Receber confiança; Observar contexto

Fundamentação Teórica:
    Este agente implementa um sistema de tomada de decisão baseado em regras
    que considera não apenas o sentimento detectado, mas também a confiabilidade
    dessa predição. Isso demonstra coordenação real entre agentes, onde a
    decisão de um agente depende da saída de outro.
    
    Referências:
    - Russell, S. & Norvig, P. (2020). Artificial Intelligence: A Modern Approach
"""

from typing import Set, Dict, Any, Optional, Tuple
from .base_agent import BaseAgent, PEAS, AgentPercept, Performative


class ActionAgent(BaseAgent):
    """
    Agente especializado em recomendação de ações táticas.
    
    Implementa regras de negócio para determinar qual ação tomar
    com base no sentimento identificado e no status de validação.
    
    Comportamento autônomo:
    - Considera múltiplos fatores na decisão (não só sentimento)
    - Escala automaticamente casos de baixa confiança
    - Aprende padrões de ação bem-sucedidas (histórico)
    
    Attributes:
        HUMAN_REVIEW_STATUSES: Status que requerem intervenção humana
        ACTION_PRIORITY: Mapeamento de prioridade por tipo de ação
    """
    
    # Status que requerem intervenção humana
    HUMAN_REVIEW_STATUSES: Set[str] = {"BAIXA_CONFIANCA", "AMBIGUO", "OOD"}
    
    # Prioridade das ações (para métricas)
    ACTION_PRIORITY = {
        "escalar_humano": 1,      # Máxima prioridade
        "atender_urgente": 2,
        "atender_cauteloso": 3,
        "agradecer_supervisionado": 4,
        "agradecer_automatico": 5,
        "registrar_neutro": 6,
        "registrar_posterior": 7   # Mínima prioridade
    }
    
    def __init__(self, name: str = "ActionAgent"):
        """
        Inicializa o agente de ação.
        
        Args:
            name: Identificador do agente
        """
        super().__init__(name)
        
        # Objetivos do agente
        self.goals = [
            "Maximizar satisfação do cliente",
            "Minimizar tempo de resposta para casos críticos",
            "Otimizar alocação de recursos humanos",
            "Garantir que casos de baixa confiança sejam revisados"
        ]
    
    @property
    def peas(self) -> PEAS:
        """Especificação PEAS do agente de ação."""
        return PEAS(
            performance_measures=[
                "Taxa de resolução de casos negativos",
                "Tempo médio de resposta por prioridade",
                "Taxa de escalação para humanos (target: < 20%)",
                "Satisfação do cliente pós-ação"
            ],
            environment_description=(
                "Sentimento classificado pelo SentimentAgent. "
                "Status de validação do ValidationAgent. "
                "Regras de negócio da empresa. "
                "Histórico de ações e resultados."
            ),
            actuators=[
                "Emitir ação recomendada com prioridade",
                "Escalar caso para revisão humana",
                "Priorizar atendimento urgente",
                "Registrar para análise posterior"
            ],
            sensors=[
                "Receber sentimento classificado",
                "Receber status de validação",
                "Observar confiança da predição",
                "Consultar histórico de ações similares"
            ]
        )
    
    def _initialize_beliefs(self) -> None:
        """Inicializa crenças específicas do agente de ação."""
        super()._initialize_beliefs()
        self.beliefs.update({
            "total_decisions": 0,
            "escalations_to_human": 0,
            "urgent_cases": 0,
            "automatic_responses": 0,
            "action_distribution": {},
            "average_priority": 0.0
        })
    
    def perceive(self, percept: AgentPercept) -> None:
        """
        Processa percepção e atualiza crenças sobre o caso.
        
        Args:
            percept: Percepção contendo sentimento e validação
        """
        data = percept.data
        
        sentiment = data.get("sentiment", "")
        validation_status = data.get("validation_status", "")
        confidence = data.get("confidence", 0.0)
        
        # Atualizar crenças sobre o caso atual
        self.beliefs["current_sentiment"] = sentiment
        self.beliefs["current_validation_status"] = validation_status
        self.beliefs["current_confidence"] = confidence
        
        # Determinar se requer revisão humana
        self.beliefs["requires_human_review"] = validation_status in self.HUMAN_REVIEW_STATUSES
        
        # Determinar urgência
        self.beliefs["is_urgent"] = (
            sentiment == "Negativo" and 
            validation_status == "CONFIAVEL"
        )
        
        # Determinar se pode ser automático
        self.beliefs["can_be_automatic"] = (
            sentiment == "Positivo" and 
            validation_status == "CONFIAVEL" and
            confidence >= 0.75
        )
    
    def can_handle(self, request: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Verifica se pode processar a requisição.
        
        Args:
            request: Requisição com sentimento e validação
            
        Returns:
            Tupla (pode_processar, motivo)
        """
        sentiment = request.get("sentiment", "")
        validation_status = request.get("validation_status", "")
        
        if not sentiment:
            return False, "Sentimento não fornecido"
        
        if not validation_status:
            return False, "Status de validação não fornecido"
        
        valid_sentiments = {"Positivo", "Neutro", "Negativo"}
        if sentiment not in valid_sentiments:
            return False, f"Sentimento inválido: {sentiment}"
        
        return True, "Requisição aceita"
    
    def decide(self) -> Optional[str]:
        """
        Decide qual ação executar baseado nas crenças.
        
        Hierarquia de decisão:
        1. Baixa confiança → escalar para humano
        2. Negativo + confiável → atendimento urgente
        3. Negativo + moderado → atendimento cauteloso
        4. Positivo + confiável → agradecimento automático
        5. Positivo + moderado → agradecimento supervisionado
        6. Neutro → registrar
        
        Returns:
            Ação a executar
        """
        # 1. Casos de baixa confiança sempre escalados
        if self.beliefs.get("requires_human_review", False):
            return "escalar_humano"
        
        sentiment = self.beliefs.get("current_sentiment", "")
        validation_status = self.beliefs.get("current_validation_status", "")
        
        # 2. Sentimento negativo
        if sentiment == "Negativo":
            if validation_status == "CONFIAVEL":
                return "atender_urgente"
            else:
                return "atender_cauteloso"
        
        # 3. Sentimento positivo
        if sentiment == "Positivo":
            if self.beliefs.get("can_be_automatic", False):
                return "agradecer_automatico"
            else:
                return "agradecer_supervisionado"
        
        # 4. Neutro ou outros
        if validation_status == "CONFIAVEL":
            return "registrar_neutro"
        
        return "registrar_posterior"
    
    def act(self, action: str) -> Dict[str, Any]:
        """
        Executa a ação e gera recomendação.
        
        Args:
            action: Tipo de ação a executar
            
        Returns:
            Resultado com ação recomendada
        """
        self.beliefs["total_decisions"] = self.beliefs.get("total_decisions", 0) + 1
        
        # Mapear ação para mensagem
        action_messages = {
            "escalar_humano": "Encaminhar para revisão humana - predição requer validação manual.",
            "atender_urgente": "Priorizar atendimento - cliente insatisfeito confirmado.",
            "atender_cauteloso": "Atender com cautela - possível insatisfação detectada.",
            "agradecer_automatico": "Agradecer automaticamente e incentivar novas compras.",
            "agradecer_supervisionado": "Agradecer com supervisão posterior.",
            "registrar_neutro": "Registrar feedback neutro e monitorar padrões.",
            "registrar_posterior": "Registrar para análise posterior."
        }
        
        message = action_messages.get(action, "❓ Ação não reconhecida")
        priority = self.ACTION_PRIORITY.get(action, 7)
        
        # Atualizar estatísticas
        if action == "escalar_humano":
            self.beliefs["escalations_to_human"] = self.beliefs.get("escalations_to_human", 0) + 1
            # Notificar outros agentes
            self.send_message(
                receiver="ResponseAgent",
                performative=Performative.INFORM,
                content={
                    "action": action,
                    "requires_human": True,
                    "priority": priority
                }
            )
        elif action == "atender_urgente":
            self.beliefs["urgent_cases"] = self.beliefs.get("urgent_cases", 0) + 1
        elif action == "agradecer_automatico":
            self.beliefs["automatic_responses"] = self.beliefs.get("automatic_responses", 0) + 1
        
        # Atualizar distribuição de ações
        dist = self.beliefs.get("action_distribution", {})
        dist[action] = dist.get(action, 0) + 1
        self.beliefs["action_distribution"] = dist
        
        # Atualizar média de prioridade
        n = self.beliefs["total_decisions"]
        old_avg = self.beliefs.get("average_priority", 0)
        self.beliefs["average_priority"] = old_avg + (priority - old_avg) / n
        
        result = {
            "success": True,
            "action": action,
            "message": message,
            "priority": priority,
            "requires_human": action == "escalar_humano",
            "is_urgent": action == "atender_urgente",
            "is_automatic": action == "agradecer_automatico"
        }
        
        return result
    
    def get_action(self, sentiment: str, validation_status: str) -> str:
        """
        Interface de alto nível para obter ação.
        
        Executa o ciclo completo do agente: perceive → decide → act.
        Mantém compatibilidade com a API anterior.
        
        Args:
            sentiment: Sentimento identificado pelo SentimentAgent
            validation_status: Status de confiabilidade do ValidationAgent
            
        Returns:
            String com a ação recomendada
        """
        # Verificar se pode processar
        can_process, reason = self.can_handle({
            "sentiment": sentiment,
            "validation_status": validation_status
        })
        
        if not can_process:
            return f"❓ Não foi possível determinar ação: {reason}"
        
        # Executar ciclo do agente
        percept = AgentPercept(
            source="ValidationAgent",
            data={
                "sentiment": sentiment,
                "validation_status": validation_status
            }
        )
        
        result = self.run_cycle(percept)
        return result.get("message", "❓ Ação não determinada")
    
    def get_action_with_details(self, sentiment: str, validation_status: str, 
                                 confidence: float = 0.0) -> Dict[str, Any]:
        """
        Obtém ação com detalhes completos.
        
        Args:
            sentiment: Sentimento identificado
            validation_status: Status de validação
            confidence: Confiança da predição
            
        Returns:
            Dicionário com ação e metadados
        """
        can_process, reason = self.can_handle({
            "sentiment": sentiment,
            "validation_status": validation_status
        })
        
        if not can_process:
            return {
                "success": False,
                "reason": reason
            }
        
        percept = AgentPercept(
            source="ValidationAgent",
            data={
                "sentiment": sentiment,
                "validation_status": validation_status,
                "confidence": confidence
            }
        )
        
        return self.run_cycle(percept)
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de performance do agente."""
        total = self.beliefs.get("total_decisions", 0)
        return {
            "total_decisions": total,
            "escalations_to_human": self.beliefs.get("escalations_to_human", 0),
            "urgent_cases": self.beliefs.get("urgent_cases", 0),
            "automatic_responses": self.beliefs.get("automatic_responses", 0),
            "escalation_rate": self.beliefs.get("escalations_to_human", 0) / max(total, 1),
            "average_priority": self.beliefs.get("average_priority", 0.0),
            "action_distribution": self.beliefs.get("action_distribution", {})
        }
