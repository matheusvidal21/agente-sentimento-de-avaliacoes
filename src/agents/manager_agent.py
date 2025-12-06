"""
Agente Gerenciador (Manager Agent).

Responsável por orquestrar todos os agentes especializados e
coordenar o fluxo completo de análise de avaliações.

Arquitetura Multi-Agente:
    O ManagerAgent implementa o padrão de orquestração centralizada,
    coordenando a execução sequencial dos agentes especializados:
    
    Sentimento → Validação → Keywords → Ação → Resposta
    
    A separação de responsabilidades justifica a arquitetura multi-agente:
    - SentimentAgent: Especialista em classificação
    - ValidationAgent: Especialista em quantificação de incerteza
    - KeywordAgent: Especialista em extração de features
    - ActionAgent: Especialista em tomada de decisão
    - ResponseAgent: Especialista em geração de texto
"""

import time
from typing import Dict, Any
import joblib
from ..model_persistence import NB_MODEL_PATH, LR_MODEL_PATH, VECTORIZER_PATH
from .sentiment_agent import SentimentAgent
from .validation_agent import ValidationAgent
from .keyword_agent import KeywordAgent
from .action_agent import ActionAgent
from .response_agent import ResponseAgent


class ManagerAgent:
    """
    Agente coordenador do sistema multi-agente.
    
    Gerencia a carga de modelos e orquestra o pipeline completo:
    Sentimento → Validação → Keywords → Ação → Resposta
    
    A arquitetura demonstra coordenação real entre agentes:
    - ValidationAgent avalia a confiabilidade do SentimentAgent
    - ActionAgent decide baseado no status do ValidationAgent
    - ResponseAgent adapta tom baseado na validação
    """
    
    def __init__(self):
        """
        Inicializa o gerenciador e carrega todos os artefatos necessários.
        """
        self.load_artifacts()
        
        self.sentiment_agents = {
            "nb": SentimentAgent(self.nb_model, self.vectorizer),
            "lr": SentimentAgent(self.lr_model, self.vectorizer)
        }
        self.validation_agent = ValidationAgent()
        self.keyword_agent = KeywordAgent(self.vectorizer)
        self.action_agent = ActionAgent()
        self.response_agent = ResponseAgent()

    def load_artifacts(self) -> None:
        """
        Carrega modelos e vetorizador do disco.
        
        Raises:
            FileNotFoundError: Se algum arquivo de modelo não for encontrado
        """
        try:
            self.vectorizer = joblib.load(VECTORIZER_PATH)
            self.nb_model = joblib.load(NB_MODEL_PATH)
            self.lr_model = joblib.load(LR_MODEL_PATH)
        except FileNotFoundError as e:
            print(f"Erro ao carregar artefatos: {e}")
            raise e

    def process(self, text: str, model_type: str = "lr") -> Dict[str, Any]:
        """
        Executa o pipeline completo de análise.
        
        Orquestra todos os agentes especializados para analisar uma avaliação,
        gerando métricas de execução e trace detalhado do processo.
        
        Args:
            text: Texto da avaliação a ser analisada
            model_type: Tipo de modelo de sentimento ("nb" ou "lr")
            
        Returns:
            Dicionário com resultados consolidados e trace de execução
        """
        execution_trace = []
        total_start = time.time()

        sentiment_agent = self.sentiment_agents.get(model_type)
        if not sentiment_agent:
            return {"error": f"Tipo de modelo '{model_type}' desconhecido. Use 'nb' ou 'lr'."}

        # 1. Análise de Sentimento
        start_time = time.time()
        sentiment_result = sentiment_agent.predict(text)
        execution_time = (time.time() - start_time) * 1000
        
        conf = sentiment_result['probabilities'][sentiment_result['label']]
        probs_formatted = {k: f"{float(v):.2%}" for k, v in sentiment_result['probabilities'].items()}
        
        execution_trace.append({
            "agent": "Agente de Sentimento",
            "icon": "analysis",
            "summary": f"Classificação: {sentiment_result['label']}",
            "details": f"Modelo {model_type.upper()} calculou probabilidades de sentimento.\nConfiança: {conf:.1%}\nProbabilidades: {probs_formatted}",
            "execution_time_ms": round(execution_time, 2)
        })
        
        # 2. Validação de Confiabilidade
        start_time = time.time()
        validation_result = self.validation_agent.validate(text, sentiment_result, model_type)
        execution_time = (time.time() - start_time) * 1000
        
        status_emoji = {
            "CONFIAVEL": "✅",
            "CONFIANCA_MODERADA": "⚠️",
            "BAIXA_CONFIANCA": "❌",
            "AMBIGUO": "🔀",
            "OOD": "⚡"
        }.get(validation_result["status"], "❓")
        
        execution_trace.append({
            "agent": "Agente de Validação",
            "icon": "verified",
            "summary": f"{status_emoji} Status: {validation_result['status']}",
            "details": f"Confiança: {validation_result['confianca']:.1%} | Entropia: {validation_result['entropia']:.3f} bits\n{validation_result['recomendacao']}",
            "execution_time_ms": round(execution_time, 2)
        })
        
        # 3. Extração de Palavras-Chave
        start_time = time.time()
        keywords = self.keyword_agent.extract_keywords(text)
        execution_time = (time.time() - start_time) * 1000
        
        execution_trace.append({
            "agent": "Agente de Palavras-Chave",
            "icon": "search",
            "summary": "Extração de Tópicos",
            "details": f"TF-IDF identificou termos mais relevantes.\nTop termos: {', '.join(keywords)}",
            "execution_time_ms": round(execution_time, 2)
        })
        
        explanation = sentiment_agent.explain(text)
        
        # 4. Definição de Ação
        start_time = time.time()
        action = self.action_agent.get_action(sentiment_result['label'], validation_result['status'])
        execution_time = (time.time() - start_time) * 1000
        
        # Determinar risco baseado em sentimento + validação
        if validation_result['requer_revisao_humana']:
            risk_score = "Incerto - Requer Revisão"
        else:
            risk_score = {
                "Negativo": "Alto",
                "Neutro": "Médio",
                "Positivo": "Baixo"
            }.get(sentiment_result['label'], "Baixo")
        
        execution_trace.append({
            "agent": "Agente de Ação",
            "icon": "gavel",
            "summary": "Decisão Tática",
            "details": f"Risco: '{risk_score}' | Validação: {validation_result['status']}\nAção: {action}",
            "execution_time_ms": round(execution_time, 2)
        })

        # 5. Geração de Resposta
        start_time = time.time()
        generated_reply = self.response_agent.generate_reply(
            text, 
            sentiment_result['label'], 
            validation_result, 
            action
        )
        execution_time = (time.time() - start_time) * 1000
        
        execution_trace.append({
            "agent": "Agente de Resposta",
            "icon": "chat",
            "summary": "Geração Criativa",
            "details": "Gemini 2.0 Flash gerou resposta empática e contextualizada.",
            "execution_time_ms": round(execution_time, 2)
        })
        
        total_time = (time.time() - total_start) * 1000
        
        return {
            "text": text,
            "model_type": model_type,
            "sentiment_analysis": sentiment_result,
            "validation": validation_result,
            "keywords": keywords,
            "explanation": explanation,
            "suggested_action": action,
            "generated_reply": generated_reply,
            "execution_trace": execution_trace,
            "total_execution_time_ms": round(total_time, 2)
        }
