"""
Agente Gerenciador (Manager Agent).

Responsável por orquestrar todos os agentes especializados e
coordenar o fluxo completo de análise de avaliações.

Especificação PEAS:
    Performance: Minimizar latência; Maximizar qualidade; Coordenar eficientemente
    Environment: Estados dos agentes; Fila de avaliações; Recursos disponíveis
    Actuators: Orquestrar pipeline; Paralelizar; Replanejar se necessário
    Sensors: Monitorar execução; Receber status; Detectar falhas

Arquitetura Multi-Agente (Ensemble):
    O ManagerAgent implementa o padrão de orquestração centralizada,
    coordenando a execução dos agentes especializados usando SEMPRE
    o modo ensemble: ambos os modelos (Naive Bayes e Regressão Logística)
    são executados e o ValidationAgent decide qual resultado é mais confiável.
    
    Pipeline:
    [NB + LR] → Validação (escolhe melhor) → Explicabilidade → Ação → Resposta
    
    Características de coordenação real:
    - Ensemble: usa ambos os modelos e ValidationAgent arbitra
    - Replanning: se ValidationAgent indica baixa confiança, adapta fluxo
    - Comunicação: processa mensagens entre agentes
    - Monitoramento: rastreia performance de cada agente

Referências:
    - Russell, S. & Norvig, P. (2020). Artificial Intelligence: A Modern Approach
    - Wooldridge, M. (2009). An Introduction to MultiAgent Systems
"""

import time
from typing import Dict, Any, List, Optional
import joblib

from .base_agent import BaseAgent, PEAS, AgentPercept, Performative, AgentMessage
from ..model_persistence import NB_MODEL_PATH, LR_MODEL_PATH, VECTORIZER_PATH
from .sentiment_agent import SentimentAgent
from .validation_agent import ValidationAgent
from .explainability_agent import ExplainabilityAgent
from .action_agent import ActionAgent
from .response_agent import ResponseAgent


class ManagerAgent(BaseAgent):
    """
    Agente coordenador do sistema multi-agente.
    
    Gerencia a carga de modelos e orquestra o pipeline completo usando
    o modo ENSEMBLE: ambos os modelos (Naive Bayes e Regressão Logística)
    são executados e o ValidationAgent decide qual resultado é mais confiável.
    
    Pipeline: [NB + LR] → Validação → Explicabilidade → Ação → Resposta
    
    Comportamento autônomo:
    - Ensemble: executa ambos os modelos e ValidationAgent arbitra
    - Replanejamento quando ValidationAgent indica problemas
    - Processamento de mensagens entre agentes
    - Coleta de estatísticas agregadas do sistema
    
    Attributes:
        sentiment_agents: Dicionário com agentes de sentimento (nb, lr)
        validation_agent: Agente de validação (também arbitra entre modelos)
        explainability_agents: Agentes de explicabilidade (um por modelo)
        action_agent: Agente de decisão
        response_agent: Agente de geração de resposta
    """
    
    def __init__(self, name: str = "ManagerAgent"):
        """
        Inicializa o gerenciador e carrega todos os artefatos necessários.
        
        Args:
            name: Identificador do agente
        """
        super().__init__(name)
        self.load_artifacts()
        self._initialize_agents()
        
        # Objetivos do agente coordenador
        self.goals = [
            "Minimizar latência total do pipeline",
            "Maximizar qualidade da resposta final",
            "Coordenar agentes de forma eficiente",
            "Garantir tratamento de erros robusto"
        ]
    
    @property
    def peas(self) -> PEAS:
        """Especificação PEAS do agente gerenciador."""
        return PEAS(
            performance_measures=[
                "Latência total do pipeline (target: < 2s)",
                "Taxa de sucesso de processamento (target: > 99%)",
                "Qualidade agregada das respostas",
                "Eficiência de coordenação entre agentes"
            ],
            environment_description=(
                "Estados internos de todos os agentes especializados. "
                "Fila de avaliações para processamento. "
                "Recursos computacionais disponíveis. "
                "Mensagens pendentes entre agentes."
            ),
            actuators=[
                "Orquestrar execução sequencial do pipeline",
                "Paralelizar quando possível",
                "Interromper pipeline em caso de erro crítico",
                "Replanejar fluxo baseado em feedback dos agentes"
            ],
            sensors=[
                "Monitorar tempo de execução de cada agente",
                "Receber status e mensagens de cada agente",
                "Detectar falhas e exceções",
                "Observar métricas de performance"
            ]
        )
    
    def _initialize_beliefs(self) -> None:
        """Inicializa crenças específicas do gerenciador."""
        super()._initialize_beliefs()
        self.beliefs.update({
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_latency_ms": 0.0,
            "replanning_count": 0,
            "agents_initialized": False
        })
    
    def _initialize_agents(self) -> None:
        """Inicializa todos os agentes especializados."""
        self.sentiment_agents = {
            "nb": SentimentAgent(self.nb_model, self.vectorizer, name="SentimentAgent_NB"),
            "lr": SentimentAgent(self.lr_model, self.vectorizer, name="SentimentAgent_LR")
        }
        self.validation_agent = ValidationAgent()
        # ExplainabilityAgent para cada modelo (NB e LR têm pesos diferentes)
        self.explainability_agents = {
            "nb": ExplainabilityAgent(self.nb_model, self.vectorizer, name="ExplainabilityAgent_NB"),
            "lr": ExplainabilityAgent(self.lr_model, self.vectorizer, name="ExplainabilityAgent_LR")
        }
        self.action_agent = ActionAgent()
        self.response_agent = ResponseAgent()
        
        self.beliefs["agents_initialized"] = True

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
            self.alert(f"Erro ao carregar artefatos: {e}", severity="error")
            raise e
    
    def perceive(self, percept: AgentPercept) -> None:
        """
        Processa percepção (requisição de análise).
        
        Args:
            percept: Percepção contendo texto e configurações
        """
        data = percept.data
        
        self.beliefs["current_text"] = data.get("text", "")
        self.beliefs["request_timestamp"] = time.time()
    
    def can_handle(self, request: Dict[str, Any]) -> tuple[bool, str]:
        """
        Verifica se pode processar a requisição.
        
        Args:
            request: Requisição com texto
            
        Returns:
            Tupla (pode_processar, motivo)
        """
        text = request.get("text", "")
        
        if not text or not text.strip():
            return False, "Texto vazio"
        
        if not self.beliefs.get("agents_initialized", False):
            return False, "Agentes não inicializados"
        
        return True, "Requisição aceita"
    
    def decide(self) -> Optional[str]:
        """
        Decide qual pipeline executar.
        
        Returns:
            Ação a executar
        """
        # Pipeline padrão
        return "execute_pipeline"
    
    def act(self, action: str) -> Dict[str, Any]:
        """
        Executa o pipeline completo usando ambos os modelos (ensemble).
        
        O ValidationAgent compara as predições de Naive Bayes e Regressão
        Logística e escolhe o resultado mais confiável.
        
        Args:
            action: Ação a executar
            
        Returns:
            Resultado consolidado do pipeline
        """
        if action != "execute_pipeline":
            return {"error": f"Ação desconhecida: {action}"}
        
        text = self.beliefs.get("current_text", "")
        
        # Sempre usa ensemble: ambos os modelos são executados e
        # o ValidationAgent decide qual resultado é mais confiável
        return self._execute_ensemble_pipeline(text)
    
    def _execute_pipeline(self, text: str, model_type: str) -> Dict[str, Any]:
        """
        Executa o pipeline de análise completo.
        
        Args:
            text: Texto a analisar
            model_type: Tipo de modelo (nb ou lr)
            
        Returns:
            Resultado consolidado
        """
        execution_trace = []
        total_start = time.time()
        
        self.beliefs["total_requests"] = self.beliefs.get("total_requests", 0) + 1
        
        sentiment_agent = self.sentiment_agents.get(model_type)

        # 1. Análise de Sentimento
        start_time = time.time()
        sentiment_result = sentiment_agent.predict(text)
        execution_time = (time.time() - start_time) * 1000
        
        # Processar mensagens do SentimentAgent
        self._process_agent_messages(sentiment_agent)
        
        # Verificar se foi recusado
        if sentiment_result.get("refused", False):
            return self._handle_refused_request(sentiment_result, execution_trace, total_start)
        
        conf = sentiment_result['probabilities'][sentiment_result['label']]
        probs_formatted = {k: f"{float(v):.2%}" for k, v in sentiment_result['probabilities'].items()}
        
        execution_trace.append({
            "agent": "Agente de Sentimento",
            "icon": "analysis",
            "summary": f"Classificação: {sentiment_result['label']}",
            "details": (
                f"Modelo {model_type.upper()} calculou probabilidades de sentimento.\n"
                f"Confiança: {conf:.1%}\n"
                f"Probabilidades: {probs_formatted}"
            ),
            "execution_time_ms": round(execution_time, 2),
            "agent_stats": sentiment_agent.get_agent_stats()
        })
        
        # 2. Validação de Confiabilidade
        start_time = time.time()
        validation_result = self.validation_agent.validate(text, sentiment_result, model_type)
        execution_time = (time.time() - start_time) * 1000
        
        # Processar mensagens do ValidationAgent
        self._process_agent_messages(self.validation_agent)
        
        # Status visual mais claro
        status_display = validation_result['status'].replace('_', ' ').upper()
        
        execution_trace.append({
            "agent": "Agente de Validação",
            "icon": "verified",
            "summary": f"Status: {status_display} | Confiança: {validation_result['confianca']:.0%}",
            "details": (
                f"VALIDAÇÃO DO RESULTADO:\n"
                f"Status: {status_display}\n"
                f"Confiança: {validation_result['confianca']:.1%}\n"
                f"Entropia: {validation_result['entropia']:.3f} bits\n"
                f"Entropia Normalizada: {validation_result['entropia_normalizada']:.1%}\n\n"
                f"RECOMENDAÇÃO:\n"
                f"{validation_result['recomendacao']}"
            ),
            "execution_time_ms": round(execution_time, 2)
        })
        
        # 3. Explicabilidade da Predição (XAI)
        start_time = time.time()
        explainability_agent = self.explainability_agents[model_type]
        explainability_result = explainability_agent.explain_prediction(
            text, 
            sentiment_result['label']
        )
        execution_time = (time.time() - start_time) * 1000
        
        # Extrair keywords das palavras mais influentes para compatibilidade
        keywords = []
        keyword_scores = {}
        for word, score in explainability_result.get("palavras_positivas", []):
            keywords.append(word)
            keyword_scores[word] = score
        for word, score in explainability_result.get("palavras_negativas", []):
            keywords.append(word)
            keyword_scores[word] = score
        
        # Processar mensagens do ExplainabilityAgent
        self._process_agent_messages(explainability_agent)
        
        execution_trace.append({
            "agent": "Agente de Explicabilidade",
            "icon": "lightbulb",
            "summary": "Análise XAI",
            "details": (
                f"EXPLICAÇÃO DA PREDIÇÃO:\n"
                f"{explainability_result.get('explicacao', '')}\n\n"
                f"Palavras positivas: {', '.join([w for w, _ in explainability_result.get('palavras_positivas', [])])}\n"
                f"Palavras negativas: {', '.join([w for w, _ in explainability_result.get('palavras_negativas', [])])}"
            ),
            "execution_time_ms": round(execution_time, 2)
        })
        
        # Explicação do modelo
        explanation = sentiment_agent.explain(text)
        
        # 4. Definição de Ação
        start_time = time.time()
        action = self.action_agent.get_action(sentiment_result['label'], validation_result['status'])
        execution_time = (time.time() - start_time) * 1000
        
        # Processar mensagens do ActionAgent
        self._process_agent_messages(self.action_agent)
        
        # Determinar risco
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
        
        # Processar mensagens do ResponseAgent
        self._process_agent_messages(self.response_agent)
        
        execution_trace.append({
            "agent": "Agente de Resposta",
            "icon": "chat",
            "summary": "Geração Criativa",
            "details": "LLM gerou resposta empática e contextualizada.",
            "execution_time_ms": round(execution_time, 2)
        })
        
        total_time = (time.time() - total_start) * 1000
        
        # Atualizar estatísticas
        self.beliefs["successful_requests"] = self.beliefs.get("successful_requests", 0) + 1
        n = self.beliefs["successful_requests"]
        old_avg = self.beliefs.get("average_latency_ms", 0)
        self.beliefs["average_latency_ms"] = old_avg + (total_time - old_avg) / n
        
        return {
            "text": text,
            "model_type": model_type,
            "sentiment_analysis": sentiment_result,
            "validation": validation_result,
            "keywords": keywords,
            "keyword_scores": keyword_scores,
            "explainability": explainability_result,
            "explanation": explanation,
            "suggested_action": action,
            "generated_reply": generated_reply,
            "execution_trace": execution_trace,
            "total_execution_time_ms": round(total_time, 2)
        }
    
    def _execute_ensemble_pipeline(self, text: str) -> Dict[str, Any]:
        """
        Executa o pipeline usando ambos os modelos e o ValidationAgent escolhe o melhor.
        
        O agente de validação compara as predições dos modelos Naive Bayes e
        Regressão Logística, ponderando entre confiança, entropia e spread
        para escolher o resultado mais confiável.
        
        Args:
            text: Texto a analisar
            
        Returns:
            Resultado consolidado com comparação entre modelos
        """
        execution_trace = []
        total_start = time.time()
        
        self.beliefs["total_requests"] = self.beliefs.get("total_requests", 0) + 1
        
        # 1. Análise de Sentimento com AMBOS os modelos
        start_time = time.time()
        
        # Executar Naive Bayes
        nb_agent = self.sentiment_agents["nb"]
        nb_result = nb_agent.predict(text)
        self._process_agent_messages(nb_agent)
        
        # Executar Regressão Logística
        lr_agent = self.sentiment_agents["lr"]
        lr_result = lr_agent.predict(text)
        self._process_agent_messages(lr_agent)
        
        execution_time = (time.time() - start_time) * 1000
        
        # Verificar se algum foi recusado
        if nb_result.get("refused", False) or lr_result.get("refused", False):
            refused_result = nb_result if nb_result.get("refused", False) else lr_result
            return self._handle_refused_request(refused_result, execution_trace, total_start)
        
        nb_conf = nb_result['probabilities'][nb_result['label']]
        lr_conf = lr_result['probabilities'][lr_result['label']]
        
        execution_trace.append({
            "agent": "Agentes de Sentimento (Ensemble)",
            "icon": "analysis",
            "summary": f"NB: {nb_result['label']} ({nb_conf:.1%}) | LR: {lr_result['label']} ({lr_conf:.1%})",
            "details": (
                f"Executados ambos os modelos em paralelo.\n"
                f"Naive Bayes: {nb_result['label']} (conf: {nb_conf:.1%})\n"
                f"Regressão Logística: {lr_result['label']} (conf: {lr_conf:.1%})"
            ),
            "execution_time_ms": round(execution_time, 2),
            "agent_stats": {
                "nb": nb_agent.get_agent_stats(),
                "lr": lr_agent.get_agent_stats()
            }
        })
        
        # 2. Agente de Validação - Compara modelos, escolhe o melhor e valida resultado
        validation_start_time = time.time()
        
        # Comparar e escolher o melhor modelo
        comparison_result = self.validation_agent.compare_predictions(text, nb_result, lr_result)
        
        chosen_model = comparison_result["chosen_model"]
        sentiment_result = comparison_result["chosen_result"]
        
        # Validar o resultado escolhido
        validation_result = self.validation_agent.validate(text, sentiment_result, chosen_model)
        
        validation_execution_time = (time.time() - validation_start_time) * 1000
        
        self._process_agent_messages(self.validation_agent)
        
        # Adicionar informação extra sobre a comparação
        validation_result["ensemble_comparison"] = comparison_result["comparison"]
        validation_result["models_agree"] = comparison_result["models_agree"]
        
        # Se modelos discordam e margem baixa, forçar revisão humana
        if comparison_result["requires_human_review"]:
            validation_result["requer_revisao_humana"] = True
            validation_result["recomendacao"] = (
                "Modelos divergem com baixa margem de decisão. "
                "Recomenda-se revisão por especialista."
            )
        
        # Status visual mais claro
        status_display = validation_result['status'].replace('_', ' ').upper()
        models_status = "Concordam" if comparison_result["models_agree"] else "Divergem"
        
        execution_trace.append({
            "agent": "Agente de Validação",
            "icon": "verified",
            "summary": (
                f"Modelo Escolhido: {chosen_model.upper()} → {sentiment_result['label']} | "
                f"Status: {status_display} | Modelos: {models_status}"
            ),
            "details": (
                f"ARBITRAGEM DE MODELOS:\n"
                f"{comparison_result['justificativa']}\n\n"
                f"VALIDAÇÃO DO RESULTADO:\n"
                f"Status: {status_display}\n"
                f"Confiança: {validation_result['confianca']:.1%}\n"
                f"Entropia: {validation_result['entropia']:.3f} bits\n"
                f"Entropia Normalizada: {validation_result['entropia_normalizada']:.1%}\n\n"
                f"RECOMENDAÇÃO:\n"
                f"{validation_result['recomendacao']}"
            ),
            "execution_time_ms": round(validation_execution_time, 2)
        })
        
        # 3. Explicabilidade da Predição (XAI)
        start_time = time.time()
        explainability_agent = self.explainability_agents[chosen_model]
        explainability_result = explainability_agent.explain_prediction(
            text, 
            sentiment_result['label']
        )
        execution_time = (time.time() - start_time) * 1000
        
        # Extrair keywords das palavras mais influentes para compatibilidade
        keywords = []
        keyword_scores = {}
        for word, score in explainability_result.get("palavras_positivas", []):
            keywords.append(word)
            keyword_scores[word] = score
        for word, score in explainability_result.get("palavras_negativas", []):
            keywords.append(word)
            keyword_scores[word] = score
        
        self._process_agent_messages(explainability_agent)
        
        execution_trace.append({
            "agent": "Agente de Explicabilidade",
            "icon": "lightbulb",
            "summary": "Análise XAI",
            "details": (
                f"EXPLICAÇÃO DA PREDIÇÃO:\n"
                f"{explainability_result.get('explicacao', '')}\n\n"
                f"Palavras positivas: {', '.join([w for w, _ in explainability_result.get('palavras_positivas', [])])}\n"
                f"Palavras negativas: {', '.join([w for w, _ in explainability_result.get('palavras_negativas', [])])}"
            ),
            "execution_time_ms": round(execution_time, 2)
        })
        
        # Explicação do modelo escolhido
        chosen_agent = self.sentiment_agents[chosen_model]
        explanation = chosen_agent.explain(text)
        
        # 4. Definição de Ação
        start_time = time.time()
        action = self.action_agent.get_action(sentiment_result['label'], validation_result['status'])
        execution_time = (time.time() - start_time) * 1000
        
        self._process_agent_messages(self.action_agent)
        
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
        
        self._process_agent_messages(self.response_agent)
        
        execution_trace.append({
            "agent": "Agente de Resposta",
            "icon": "chat",
            "summary": "Geração Criativa",
            "details": "LLM gerou resposta empática e contextualizada.",
            "execution_time_ms": round(execution_time, 2)
        })
        
        total_time = (time.time() - total_start) * 1000
        
        # Atualizar estatísticas
        self.beliefs["successful_requests"] = self.beliefs.get("successful_requests", 0) + 1
        n = self.beliefs["successful_requests"]
        old_avg = self.beliefs.get("average_latency_ms", 0)
        self.beliefs["average_latency_ms"] = old_avg + (total_time - old_avg) / n
        
        return {
            "text": text,
            "model_type": "ensemble",
            "chosen_model": chosen_model,
            "sentiment_analysis": sentiment_result,
            "ensemble_comparison": comparison_result,
            "validation": validation_result,
            "keywords": keywords,
            "keyword_scores": keyword_scores,
            "explainability": explainability_result,
            "explanation": explanation,
            "suggested_action": action,
            "generated_reply": generated_reply,
            "execution_trace": execution_trace,
            "total_execution_time_ms": round(total_time, 2)
        }
    
    def _handle_refused_request(
        self, 
        sentiment_result: Dict[str, Any],
        execution_trace: List[Dict],
        total_start: float
    ) -> Dict[str, Any]:
        """
        Trata requisições recusadas pelo SentimentAgent.
        
        Args:
            sentiment_result: Resultado com recusa
            execution_trace: Trace de execução
            total_start: Tempo de início
            
        Returns:
            Resposta de erro formatada
        """
        self.beliefs["failed_requests"] = self.beliefs.get("failed_requests", 0) + 1
        
        execution_trace.append({
            "agent": "Agente de Sentimento",
            "icon": "error",
            "summary": "Requisição Recusada",
            "details": sentiment_result.get("reason", "Motivo não especificado"),
            "execution_time_ms": 0
        })
        
        total_time = (time.time() - total_start) * 1000
        
        return {
            "error": True,
            "reason": sentiment_result.get("reason", "Requisição recusada"),
            "execution_trace": execution_trace,
            "total_execution_time_ms": round(total_time, 2)
        }
    
    def _process_agent_messages(self, agent: BaseAgent) -> None:
        """
        Processa mensagens pendentes de um agente.
        
        Args:
            agent: Agente com mensagens a processar
        """
        while agent.outbox:
            message = agent.outbox.pop(0)
            self._handle_message(message)
    
    def _handle_message(self, message: AgentMessage) -> None:
        """
        Processa uma mensagem recebida.
        
        Args:
            message: Mensagem a processar
        """
        if message.performative == Performative.ALERT:
            # Logar alertas
            severity = message.content.get("severity", "info")
            issue = message.content.get("issue", "")
            self.log_action(
                action=f"alert_from_{message.sender}",
                result={"severity": severity, "issue": issue},
                success=True
            )
        
        elif message.performative == Performative.REFUSE:
            # Registrar recusas
            reason = message.content.get("reason", "")
            self.log_action(
                action=f"refuse_from_{message.sender}",
                result={"reason": reason},
                success=False
            )
        
        elif message.performative == Performative.INFORM:
            # Processar informações (pode disparar replanejamento)
            if message.content.get("requires_human_review", False):
                self.beliefs["replanning_count"] = self.beliefs.get("replanning_count", 0) + 1

    def process(self, text: str, model_type: str = "ensemble") -> Dict[str, Any]:
        """
        Interface de alto nível para processamento.
        
        Executa o ciclo completo do agente usando o modo ensemble:
        ambos os modelos (NB e LR) são executados e o ValidationAgent
        escolhe o resultado mais confiável.
        
        Args:
            text: Texto da avaliação a ser analisada
            model_type: Parâmetro mantido para compatibilidade (ignorado, sempre usa ensemble)
            
        Returns:
            Dicionário com resultados consolidados
        """
        # Verificar se pode processar
        can_process, reason = self.can_handle({"text": text})
        
        if not can_process:
            return {"error": reason}
        
        # Executar ciclo do agente
        percept = AgentPercept(
            source="environment",
            data={"text": text}
        )
        
        return self.run_cycle(percept)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas agregadas de todo o sistema.
        
        Returns:
            Métricas de todos os agentes
        """
        stats = {
            "manager": {
                "total_requests": self.beliefs.get("total_requests", 0),
                "successful_requests": self.beliefs.get("successful_requests", 0),
                "failed_requests": self.beliefs.get("failed_requests", 0),
                "average_latency_ms": self.beliefs.get("average_latency_ms", 0),
                "replanning_count": self.beliefs.get("replanning_count", 0),
                "success_rate": (
                    self.beliefs.get("successful_requests", 0) / 
                    max(self.beliefs.get("total_requests", 0), 1)
                )
            },
            "agents": {}
        }
        
        # Coletar estatísticas de cada agente
        for name, agent in self.sentiment_agents.items():
            stats["agents"][f"sentiment_{name}"] = agent.get_agent_stats()
        
        stats["agents"]["validation"] = self.validation_agent.get_statistics()
        for name, agent in self.explainability_agents.items():
            stats["agents"][f"explainability_{name}"] = agent.get_agent_stats()
        stats["agents"]["action"] = self.action_agent.get_agent_stats()
        stats["agents"]["response"] = self.response_agent.get_agent_stats()
        
        return stats
    
    def get_all_peas(self) -> Dict[str, Dict]:
        """
        Retorna especificação PEAS de todos os agentes.
        
        Returns:
            Dicionário com PEAS de cada agente
        """
        peas_specs = {
            "manager": self.peas.to_dict()
        }
        
        for name, agent in self.sentiment_agents.items():
            peas_specs[f"sentiment_{name}"] = agent.peas.to_dict()
        
        peas_specs["validation"] = self.validation_agent.peas.to_dict()
        for name, agent in self.explainability_agents.items():
            peas_specs[f"explainability_{name}"] = agent.peas.to_dict()
        peas_specs["action"] = self.action_agent.peas.to_dict()
        peas_specs["response"] = self.response_agent.peas.to_dict()
        
        return peas_specs
