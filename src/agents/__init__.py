"""
Sistema Multi-Agente para Análise de Sentimentos.

Este pacote implementa uma arquitetura de agentes inteligentes especializados
que trabalham de forma coordenada para análise completa de avaliações de produtos.

Descrição:
    A arquitetura segue o framework PEAS (Performance, Environment, Actuators, Sensors)
    conforme definido em Russell & Norvig (2020). Cada agente é uma entidade autônoma
    que percebe seu ambiente e age de forma orientada a objetivos.

Arquitetura de Agentes:
    - BaseAgent: Classe base abstrata com interface PEAS e ciclo perceive→decide→act
    - SentimentAgent: Classificação de sentimento (NB/LR) com detecção de OOV
    - ValidationAgent: Quantificação de incerteza com auto-calibração
    - ExplainabilityAgent: Explicabilidade de predições ML (XAI) usando pesos reais do modelo
    - ActionAgent: Tomada de decisão baseada em regras de negócio
    - ResponseAgent: Geração de respostas (LLM) com fallback inteligente
    - ManagerAgent: Orquestração do pipeline multi-agente

Protocolo de Comunicação:
    Os agentes se comunicam através de AgentMessage com performativas
    baseadas em FIPA-ACL (REQUEST, INFORM, REFUSE, ALERT, etc.)

Referências:
    - Russell, S. & Norvig, P. (2020). Artificial Intelligence: A Modern Approach
    - Wooldridge, M. (2009). An Introduction to MultiAgent Systems
"""

from .base_agent import (
    BaseAgent, 
    PEAS, 
    AgentPercept, 
    AgentMessage, 
    Performative,
    AgentState
)
from .sentiment_agent import SentimentAgent
from .validation_agent import ValidationAgent
from .explainability_agent import ExplainabilityAgent
from .action_agent import ActionAgent
from .response_agent import ResponseAgent
from .manager_agent import ManagerAgent

__all__ = [
    # Classes de infraestrutura
    'BaseAgent',
    'PEAS',
    'AgentPercept',
    'AgentMessage',
    'Performative',
    'AgentState',
    # Agentes especializados
    'SentimentAgent',
    'ValidationAgent',
    'ExplainabilityAgent',
    'ActionAgent',
    'ResponseAgent',
    'ManagerAgent',
]
