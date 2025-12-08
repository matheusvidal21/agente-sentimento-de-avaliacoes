"""
Classe Base para Agentes Inteligentes.

Este módulo define a interface padrão para todos os agentes do sistema,
implementando o framework PEAS (Performance, Environment, Actuators, Sensors)
conforme definido em Russell & Norvig (2020).

Descrição:
    Um agente é qualquer coisa que pode ser vista como percebendo seu ambiente
    através de sensores e agindo nesse ambiente através de atuadores, de forma
    autônoma e orientada a objetivos.
    
    Referências:
    - Russell, S. & Norvig, P. (2020). Artificial Intelligence: A Modern Approach
    - Wooldridge, M. (2009). An Introduction to MultiAgent Systems
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime


class Performative(Enum):
    """
    Performativas de comunicação entre agentes (baseado em FIPA-ACL simplificado).
    
    Define os tipos de atos comunicativos que agentes podem trocar.
    """
    REQUEST = "request"          # Solicita que outro agente execute uma ação
    INFORM = "inform"            # Informa resultado ou fato
    REFUSE = "refuse"            # Recusa executar ação solicitada
    PROPOSE = "propose"          # Propõe uma ação alternativa
    CONFIRM = "confirm"          # Confirma recebimento/execução
    QUERY = "query"              # Consulta informação
    ALERT = "alert"              # Alerta sobre situação anômala


@dataclass
class AgentMessage:
    """
    Mensagem trocada entre agentes no sistema multi-agente.
    
    Implementa um protocolo de comunicação simplificado inspirado em FIPA-ACL,
    permitindo rastreabilidade e coordenação entre agentes.
    
    Attributes:
        sender: Nome do agente emissor
        receiver: Nome do agente destinatário
        performative: Tipo de ato comunicativo
        content: Conteúdo da mensagem (dicionário flexível)
        timestamp: Momento da criação da mensagem
        correlation_id: ID para rastrear conversações
    """
    sender: str
    receiver: str
    performative: Performative
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte mensagem para dicionário serializável."""
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "performative": self.performative.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id
        }


@dataclass
class AgentPercept:
    """
    Percepção do ambiente capturada pelos sensores do agente.
    
    Representa a entrada de informação que o agente recebe do ambiente
    antes de tomar decisões.
    
    Attributes:
        source: Origem da percepção (ambiente, outro agente, etc.)
        data: Dados percebidos
        timestamp: Momento da percepção
    """
    source: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PEAS:
    """
    Especificação PEAS do agente (Performance, Environment, Actuators, Sensors).
    
    Define formalmente as características do agente conforme framework padrão
    de Inteligência Artificial.
    
    Attributes:
        performance_measures: Métricas que definem sucesso do agente
        environment_description: Descrição do ambiente de atuação
        actuators: Lista de ações que o agente pode executar
        sensors: Lista de percepções que o agente pode capturar
    """
    performance_measures: List[str]
    environment_description: str
    actuators: List[str]
    sensors: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte especificação PEAS para dicionário."""
        return {
            "performance": self.performance_measures,
            "environment": self.environment_description,
            "actuators": self.actuators,
            "sensors": self.sensors
        }


class AgentState(Enum):
    """Estados possíveis do agente."""
    IDLE = "idle"                # Aguardando requisições
    PERCEIVING = "perceiving"    # Processando percepções
    DECIDING = "deciding"        # Deliberando sobre ação
    ACTING = "acting"            # Executando ação
    WAITING = "waiting"          # Aguardando resposta de outro agente
    ERROR = "error"              # Estado de erro


class BaseAgent(ABC):
    """
    Classe base abstrata para agentes inteligentes.
    
    Implementa o ciclo de vida padrão de um agente:
    perceive → decide → act
    
    Cada agente concreto deve implementar:
    - peas: Especificação PEAS do agente
    - perceive(): Processar percepções do ambiente
    - decide(): Deliberar sobre próxima ação
    - act(): Executar ação e produzir resultado
    
    Características de agente real implementadas:
    - Autonomia: pode recusar requisições inadequadas
    - Reatividade: responde a mudanças no ambiente
    - Proatividade: pode iniciar ações baseado em objetivos
    - Rastreabilidade: mantém histórico de ações
    
    Attributes:
        name: Identificador único do agente
        beliefs: Estado interno (crenças sobre o mundo)
        goals: Objetivos ativos do agente
        inbox: Fila de mensagens recebidas
        outbox: Fila de mensagens a enviar
        state: Estado atual do agente
        action_history: Histórico de ações executadas
    """
    
    def __init__(self, name: str):
        """
        Inicializa o agente com identificador único.
        
        Args:
            name: Nome identificador do agente
        """
        self.name = name
        self.beliefs: Dict[str, Any] = {}
        self.goals: List[str] = []
        self.inbox: List[AgentMessage] = []
        self.outbox: List[AgentMessage] = []
        self.state = AgentState.IDLE
        self.action_history: List[Dict[str, Any]] = []
        self._initialize_beliefs()
    
    @property
    @abstractmethod
    def peas(self) -> PEAS:
        """
        Retorna a especificação PEAS do agente.
        
        Cada agente concreto deve definir sua especificação PEAS,
        descrevendo métricas de performance, ambiente, atuadores e sensores.
        
        Returns:
            Objeto PEAS com a especificação completa
        """
        pass
    
    def _initialize_beliefs(self) -> None:
        """
        Inicializa crenças padrão do agente.
        
        Pode ser sobrescrito por agentes concretos para
        inicializar estado interno específico.
        """
        self.beliefs = {
            "last_action": None,
            "action_count": 0,
            "error_count": 0,
            "created_at": datetime.now().isoformat()
        }
    
    @abstractmethod
    def perceive(self, percept: AgentPercept) -> None:
        """
        Sensor: processa percepção e atualiza crenças.
        
        Este método implementa a função de percepção do agente,
        atualizando seu estado interno (crenças) com base nas
        informações recebidas do ambiente.
        
        Args:
            percept: Percepção recebida do ambiente
        """
        pass
    
    @abstractmethod
    def decide(self) -> Optional[str]:
        """
        Deliberação: decide próxima ação baseado em crenças e objetivos.
        
        Implementa o processo de tomada de decisão do agente,
        analisando o estado atual (crenças) e os objetivos para
        determinar qual ação executar.
        
        Returns:
            Nome da ação a executar, ou None se nenhuma ação necessária
        """
        pass
    
    @abstractmethod
    def act(self, action: str) -> Dict[str, Any]:
        """
        Atuador: executa ação e retorna resultado.
        
        Implementa a execução efetiva da ação decidida,
        produzindo mudanças no ambiente ou gerando saídas.
        
        Args:
            action: Nome da ação a executar
            
        Returns:
            Dicionário com resultado da ação
        """
        pass
    
    def can_handle(self, request: Dict[str, Any]) -> tuple[bool, str]:
        """
        Verifica se o agente pode lidar com a requisição.
        
        Implementa autonomia: o agente pode recusar requisições
        que não consegue processar adequadamente.
        
        Args:
            request: Dicionário com dados da requisição
            
        Returns:
            Tupla (pode_processar, motivo)
        """
        return True, "Requisição aceita"
    
    def send_message(self, receiver: str, performative: Performative, 
                     content: Dict[str, Any], correlation_id: Optional[str] = None) -> AgentMessage:
        """
        Envia mensagem para outro agente.
        
        Args:
            receiver: Nome do agente destinatário
            performative: Tipo de ato comunicativo
            content: Conteúdo da mensagem
            correlation_id: ID para rastrear conversação
            
        Returns:
            Mensagem criada e adicionada à outbox
        """
        message = AgentMessage(
            sender=self.name,
            receiver=receiver,
            performative=performative,
            content=content,
            correlation_id=correlation_id
        )
        self.outbox.append(message)
        return message
    
    def receive_message(self, message: AgentMessage) -> None:
        """
        Recebe mensagem de outro agente.
        
        Args:
            message: Mensagem recebida
        """
        self.inbox.append(message)
    
    def refuse(self, reason: str, receiver: str = "ManagerAgent") -> AgentMessage:
        """
        Recusa uma requisição com justificativa.
        
        Implementa autonomia: o agente pode recusar executar
        ações que considera inadequadas.
        
        Args:
            reason: Motivo da recusa
            receiver: Agente para notificar
            
        Returns:
            Mensagem de recusa
        """
        return self.send_message(
            receiver=receiver,
            performative=Performative.REFUSE,
            content={"reason": reason}
        )
    
    def alert(self, issue: str, severity: str = "warning", 
              receiver: str = "ManagerAgent") -> AgentMessage:
        """
        Emite alerta sobre situação anômala.
        
        Implementa proatividade: o agente pode iniciar comunicação
        quando detecta algo relevante.
        
        Args:
            issue: Descrição do problema
            severity: Nível de severidade (info, warning, error)
            receiver: Agente para notificar
            
        Returns:
            Mensagem de alerta
        """
        return self.send_message(
            receiver=receiver,
            performative=Performative.ALERT,
            content={"issue": issue, "severity": severity}
        )
    
    def log_action(self, action: str, result: Dict[str, Any], 
                   success: bool = True) -> None:
        """
        Registra ação no histórico para rastreabilidade.
        
        Args:
            action: Nome da ação executada
            result: Resultado da ação
            success: Se a ação foi bem-sucedida
        """
        self.action_history.append({
            "action": action,
            "result": result,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
        self.beliefs["last_action"] = action
        self.beliefs["action_count"] = self.beliefs.get("action_count", 0) + 1
        
        if not success:
            self.beliefs["error_count"] = self.beliefs.get("error_count", 0) + 1
    
    def run_cycle(self, percept: AgentPercept) -> Dict[str, Any]:
        """
        Executa um ciclo completo: perceive → decide → act.
        
        Este é o loop principal do agente, implementando o ciclo
        de vida padrão de agentes inteligentes.
        
        Args:
            percept: Percepção do ambiente
            
        Returns:
            Resultado da ação executada
        """
        # 1. Perceber
        self.state = AgentState.PERCEIVING
        self.perceive(percept)
        
        # 2. Decidir
        self.state = AgentState.DECIDING
        action = self.decide()
        
        if action is None:
            self.state = AgentState.IDLE
            return {"status": "no_action", "reason": "Nenhuma ação necessária"}
        
        # 3. Agir
        self.state = AgentState.ACTING
        result = self.act(action)
        
        # 4. Registrar e retornar
        self.log_action(action, result, success=result.get("success", True))
        self.state = AgentState.IDLE
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """
        Retorna status atual do agente.
        
        Returns:
            Dicionário com estado, crenças e estatísticas
        """
        return {
            "name": self.name,
            "state": self.state.value,
            "beliefs": self.beliefs,
            "pending_messages": len(self.inbox),
            "actions_executed": len(self.action_history),
            "peas": self.peas.to_dict()
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', state={self.state.value})"
