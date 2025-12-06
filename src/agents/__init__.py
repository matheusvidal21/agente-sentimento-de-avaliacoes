"""
Sistema Multi-Agente para Análise de Sentimentos.

Este pacote implementa uma arquitetura de agentes especializados que trabalham
de forma coordenada para análise completa de avaliações de produtos.

Arquitetura:
    - SentimentAgent: Classificação de sentimento (NB/LR)
    - ValidationAgent: Quantificação de incerteza e validação
    - KeywordAgent: Extração de palavras-chave (TF-IDF)
    - ActionAgent: Recomendação de ações táticas
    - ResponseAgent: Geração de respostas (LLM)
    - ManagerAgent: Orquestração do pipeline
"""

from .sentiment_agent import SentimentAgent
from .validation_agent import ValidationAgent
from .keyword_agent import KeywordAgent
from .action_agent import ActionAgent
from .response_agent import ResponseAgent
from .manager_agent import ManagerAgent

__all__ = [
    'SentimentAgent',
    'ValidationAgent',
    'KeywordAgent',
    'ActionAgent',
    'ResponseAgent',
    'ManagerAgent',
]
