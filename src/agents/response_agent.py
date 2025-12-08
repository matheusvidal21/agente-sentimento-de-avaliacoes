"""
Agente de Resposta Automática.

Responsável por gerar respostas personalizadas usando LLM (Gemini)
com base no contexto completo da avaliação analisada.

Especificação PEAS:
    Performance: Maximizar empatia percebida; Resolver problema do cliente
    Environment: Texto original; Sentimento; Confiança; LLM disponível
    Actuators: Gerar resposta; Solicitar aprovação; Escalar caso complexo
    Sensors: Receber contexto; Detectar tópicos sensíveis; Observar tom

Descrição:
    Este agente implementa geração de linguagem natural (NLG) utilizando
    um modelo de linguagem grande (LLM). O agente demonstra autonomia ao
    decidir entre respostas automáticas ou fallback, e ao adaptar o tom
    baseado no contexto.
    
    Referências:
    - Russell, S. & Norvig, P. (2020). Artificial Intelligence: A Modern Approach
"""

import os
from typing import Optional, Dict, Any, Tuple
from .base_agent import BaseAgent, PEAS, AgentPercept, Performative

try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class ResponseAgent(BaseAgent):
    """
    Agente especializado em geração de respostas automáticas.
    
    Utiliza o modelo Gemini para criar respostas empáticas
    e contextualizadas para avaliações de clientes.
    
    Comportamento autônomo:
    - Escolhe entre resposta LLM ou fallback baseado na disponibilidade
    - Adapta tom e conteúdo baseado na confiança da predição
    - Detecta casos que requerem revisão humana
    
    Attributes:
        model: Modelo Gemini configurado (ou None se indisponível)
        FALLBACK_RESPONSES: Respostas padrão por sentimento
    """
    
    # Respostas fallback por sentimento
    FALLBACK_RESPONSES = {
        "Positivo": (
            "Muito obrigado pelo seu feedback positivo! "
            "Ficamos felizes em saber que você teve uma boa experiência. "
            "Conte sempre conosco!"
        ),
        "Negativo": (
            "Lamentamos muito pela sua experiência negativa. "
            "Pedimos sinceras desculpas e vamos trabalhar para resolver isso. "
            "Por favor, entre em contato com nosso suporte para que possamos ajudá-lo."
        ),
        "Neutro": (
            "Obrigado pelo seu feedback! "
            "Valorizamos sua opinião e estamos sempre buscando melhorar. "
            "Se tiver alguma sugestão, estamos à disposição!"
        ),
        "Revisao": (
            "Obrigado pelo seu feedback! "
            "Um membro da nossa equipe irá analisar sua avaliação "
            "e entrar em contato em breve para melhor atendê-lo."
        )
    }
    
    def __init__(self, name: str = "ResponseAgent"):
        """
        Inicializa o agente de resposta e configura a API do Gemini.
        
        Args:
            name: Identificador do agente
        """
        super().__init__(name)
        self.model = None
        self._initialize_llm()
        
        # Objetivos do agente
        self.goals = [
            "Maximizar empatia percebida nas respostas",
            "Manter tom consistente com a marca",
            "Resolver ou encaminhar problemas do cliente",
            "Adaptar resposta ao nível de confiança"
        ]
    
    def _initialize_llm(self) -> None:
        """Inicializa o modelo LLM se disponível."""
        if not GENAI_AVAILABLE:
            self.beliefs["llm_available"] = False
            return
        
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        
        if api_key:
            try:
                genai.configure(api_key=api_key)
                
                generation_config = {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 300,
                }
                
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
                
                self.model = genai.GenerativeModel(
                    model_name="gemini-2.0-flash",
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                self.beliefs["llm_available"] = True
            except Exception as e:
                self.beliefs["llm_available"] = False
                self.beliefs["llm_error"] = str(e)
        else:
            self.beliefs["llm_available"] = False
    
    @property
    def peas(self) -> PEAS:
        """Especificação PEAS do agente de resposta."""
        return PEAS(
            performance_measures=[
                "Empatia percebida nas respostas (survey)",
                "Taxa de resolução de problemas",
                "Consistência de tom com a marca",
                "Taxa de uso de LLM vs fallback"
            ],
            environment_description=(
                "Texto original da avaliação do cliente. "
                "Sentimento classificado e confiança. "
                "Ação recomendada pelo ActionAgent. "
                "LLM (Gemini) disponível ou não."
            ),
            actuators=[
                "Gerar resposta via LLM",
                "Gerar resposta fallback",
                "Solicitar aprovação humana",
                "Escalar caso complexo"
            ],
            sensors=[
                "Receber texto e contexto completo",
                "Observar disponibilidade do LLM",
                "Detectar tópicos sensíveis no texto",
                "Receber status de validação"
            ]
        )
    
    def _initialize_beliefs(self) -> None:
        """Inicializa crenças específicas do agente de resposta."""
        super()._initialize_beliefs()
        self.beliefs.update({
            "total_responses": 0,
            "llm_responses": 0,
            "fallback_responses": 0,
            "human_review_responses": 0,
            "llm_errors": 0,
            "average_response_length": 0.0
        })
    
    def perceive(self, percept: AgentPercept) -> None:
        """
        Processa percepção e atualiza crenças sobre o contexto.
        
        Args:
            percept: Percepção contendo texto e metadados
        """
        data = percept.data
        
        text = data.get("text", "")
        sentiment = data.get("sentiment", "")
        validation_result = data.get("validation_result", {})
        action = data.get("action", "")
        
        # Extrair informações de validação
        confianca = validation_result.get("confianca", 0.5)
        status = validation_result.get("status", "DESCONHECIDO")
        requer_revisao = validation_result.get("requer_revisao_humana", False)
        
        # Atualizar crenças
        self.beliefs["current_text"] = text
        self.beliefs["current_sentiment"] = sentiment
        self.beliefs["current_confidence"] = confianca
        self.beliefs["current_status"] = status
        self.beliefs["current_action"] = action
        self.beliefs["requires_human_review"] = requer_revisao
        
        # Determinar estratégia de resposta
        self.beliefs["use_llm"] = (
            self.beliefs.get("llm_available", False) and 
            not requer_revisao
        )
        
        # Detectar casos sensíveis (exemplo simples)
        sensitive_keywords = ["reembolso", "advogado", "procon", "processo", "fraude"]
        text_lower = text.lower()
        is_sensitive = any(kw in text_lower for kw in sensitive_keywords)
        self.beliefs["is_sensitive"] = is_sensitive
    
    def can_handle(self, request: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Verifica se pode gerar resposta para a requisição.
        
        Args:
            request: Requisição com contexto
            
        Returns:
            Tupla (pode_processar, motivo)
        """
        text = request.get("text", "")
        sentiment = request.get("sentiment", "")
        
        if not text:
            return False, "Texto não fornecido"
        
        if not sentiment:
            return False, "Sentimento não fornecido"
        
        return True, "Requisição aceita"
    
    def decide(self) -> Optional[str]:
        """
        Decide qual estratégia de geração usar.
        
        Returns:
            Estratégia de geração
        """
        # Se requer revisão humana, usar resposta específica
        if self.beliefs.get("requires_human_review", False):
            return "generate_human_review"
        
        # Se caso sensível, usar fallback com cuidado
        if self.beliefs.get("is_sensitive", False):
            self.alert(
                issue="Caso sensível detectado - usando resposta cautelosa",
                severity="warning"
            )
            return "generate_fallback"
        
        # Se LLM disponível, usar
        if self.beliefs.get("use_llm", False):
            return "generate_llm"
        
        # Fallback
        return "generate_fallback"
    
    def act(self, action: str) -> Dict[str, Any]:
        """
        Executa a geração de resposta.
        
        Args:
            action: Estratégia de geração
            
        Returns:
            Resultado com resposta gerada
        """
        self.beliefs["total_responses"] = self.beliefs.get("total_responses", 0) + 1
        
        sentiment = self.beliefs.get("current_sentiment", "Neutro")
        
        if action == "generate_human_review":
            self.beliefs["human_review_responses"] = self.beliefs.get("human_review_responses", 0) + 1
            response = self.FALLBACK_RESPONSES.get("Revisao", "")
            
            # Notificar que precisa de revisão
            self.send_message(
                receiver="ManagerAgent",
                performative=Performative.INFORM,
                content={
                    "response_type": "human_review",
                    "requires_approval": True
                }
            )
            
            return {
                "success": True,
                "action": action,
                "response": response,
                "source": "fallback_human_review",
                "requires_approval": True
            }
        
        if action == "generate_llm":
            response = self._generate_llm_response()
            if response:
                self.beliefs["llm_responses"] = self.beliefs.get("llm_responses", 0) + 1
                self._update_average_length(len(response))
                return {
                    "success": True,
                    "action": action,
                    "response": response,
                    "source": "llm",
                    "requires_approval": False
                }
            else:
                # Fallback se LLM falhar
                self.beliefs["llm_errors"] = self.beliefs.get("llm_errors", 0) + 1
                action = "generate_fallback"
        
        # Fallback
        self.beliefs["fallback_responses"] = self.beliefs.get("fallback_responses", 0) + 1
        response = self.FALLBACK_RESPONSES.get(sentiment, self.FALLBACK_RESPONSES["Neutro"])
        self._update_average_length(len(response))
        
        return {
            "success": True,
            "action": action,
            "response": response,
            "source": "fallback",
            "requires_approval": False
        }
    
    def _generate_llm_response(self) -> Optional[str]:
        """
        Gera resposta usando o LLM.
        
        Returns:
            Resposta gerada ou None se falhar
        """
        if not self.model:
            return None
        
        text = self.beliefs.get("current_text", "")
        sentiment = self.beliefs.get("current_sentiment", "")
        confianca = self.beliefs.get("current_confidence", 0.5)
        status = self.beliefs.get("current_status", "")
        action = self.beliefs.get("current_action", "")
        requer_revisao = self.beliefs.get("requires_human_review", False)
        
        # Instruções baseadas na confiança
        confidence_instruction = ""
        if requer_revisao:
            confidence_instruction = (
                "- IMPORTANTE: A confiança da IA é baixa. "
                "Mencione sutilmente que um especialista revisará o caso se necessário."
            )
        elif confianca < 0.7:
            confidence_instruction = "- A confiança é moderada. Seja um pouco mais cauteloso na resposta."
        
        prompt = f"""Você é um atendente de e-commerce profissional e amigável.
Responda à avaliação do cliente de forma empática e calorosa.

Avaliação: "{text}"
Sentimento detectado: {sentiment}
Confiança da análise: {confianca:.0%}
Status de validação: {status}
Ação sugerida: {action}

Requisitos:
- Tom empático, caloroso e profissional
- Use emojis apropriados para deixar a resposta mais amigável
- Se negativo: peça desculpas sinceramente, mostre empatia e ofereça solução clara
- Se positivo: agradeça com entusiasmo e reforce o relacionamento
- Se neutro: agradeça o feedback e mostre abertura para melhorias
{confidence_instruction}

Resposta:"""
        
        try:
            response = self.model.generate_content([prompt])
            
            if not response.candidates or not response.candidates[0].content.parts:
                return None
            
            return response.text.strip()
        except Exception as e:
            self.alert(f"Erro na geração LLM: {e}", severity="error")
            return None
    
    def _update_average_length(self, length: int) -> None:
        """Atualiza média de comprimento das respostas."""
        n = self.beliefs["total_responses"]
        old_avg = self.beliefs.get("average_response_length", 0)
        self.beliefs["average_response_length"] = old_avg + (length - old_avg) / n
    
    def generate_reply(
        self, 
        text: str, 
        sentiment: str, 
        validation_result: Dict[str, Any],
        action: str
    ) -> str:
        """
        Interface de alto nível para geração de resposta.
        
        Executa o ciclo completo do agente: perceive → decide → act.
        Mantém compatibilidade com a API anterior.
        
        Args:
            text: Texto original da avaliação
            sentiment: Sentimento detectado
            validation_result: Resultado do ValidationAgent
            action: Ação sugerida pelo ActionAgent
            
        Returns:
            Resposta gerada
        """
        # Verificar se pode processar
        can_process, reason = self.can_handle({
            "text": text,
            "sentiment": sentiment
        })
        
        if not can_process:
            return self.FALLBACK_RESPONSES.get("Neutro", "Obrigado pelo seu feedback!")
        
        # Executar ciclo do agente
        percept = AgentPercept(
            source="ActionAgent",
            data={
                "text": text,
                "sentiment": sentiment,
                "validation_result": validation_result,
                "action": action
            }
        )
        
        result = self.run_cycle(percept)
        return result.get("response", self.FALLBACK_RESPONSES.get("Neutro"))
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de performance do agente."""
        total = self.beliefs.get("total_responses", 0)
        return {
            "total_responses": total,
            "llm_responses": self.beliefs.get("llm_responses", 0),
            "fallback_responses": self.beliefs.get("fallback_responses", 0),
            "human_review_responses": self.beliefs.get("human_review_responses", 0),
            "llm_errors": self.beliefs.get("llm_errors", 0),
            "llm_available": self.beliefs.get("llm_available", False),
            "llm_usage_rate": self.beliefs.get("llm_responses", 0) / max(total, 1),
            "average_response_length": self.beliefs.get("average_response_length", 0.0)
        }
