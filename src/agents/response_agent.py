"""
Agente de Resposta Automática.

Responsável por gerar respostas personalizadas usando LLM (Gemini)
com base no contexto completo da avaliação analisada.
"""

import os
from typing import Optional, Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv


class ResponseAgent:
    """
    Agente especializado em geração de respostas automáticas.
    
    Utiliza o modelo Gemini 1.5 Flash para criar respostas empáticas
    e contextualizadas para avaliações de clientes.
    """
    
    def __init__(self):
        """
        Inicializa o agente de resposta e configura a API do Gemini.
        """
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        
        if api_key:
            genai.configure(api_key=api_key)
            
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 300,
            }
            
            # Configurações de segurança mais permissivas
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
        else:
            self.model = None

    def generate_reply(
        self, 
        text: str, 
        sentiment: str, 
        validation_result: Dict[str, Any],
        action: str
    ) -> str:
        """
        Gera uma resposta automática para a avaliação.
        
        Args:
            text: Texto original da avaliação
            sentiment: Sentimento detectado (Positivo/Neutro/Negativo)
            validation_result: Resultado do ValidationAgent contendo:
                - confianca: Probabilidade da classe predita
                - status: Status de validação
                - requer_revisao_humana: Boolean
            action: Ação sugerida pelo ActionAgent
            
        Returns:
            Resposta gerada pelo LLM ou mensagem de erro
        """
        if not self.model:
            return self._generate_fallback_response(sentiment, validation_result)
        
        confianca = validation_result.get('confianca', 0.5)
        status = validation_result.get('status', 'DESCONHECIDO')
        requer_revisao = validation_result.get('requer_revisao_humana', False)
        
        # Instruções adicionais baseadas na confiança
        confidence_instruction = ""
        if requer_revisao:
            confidence_instruction = "- IMPORTANTE: A confiança da IA é baixa. Mencione sutilmente que um especialista revisará o caso se necessário."
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
            
            # Verificar se a resposta foi bloqueada por segurança
            if not response.candidates or not response.candidates[0].content.parts:
                return self._generate_fallback_response(sentiment, validation_result)
            
            return response.text.strip()
        except Exception as e:
            # Em caso de erro, retornar resposta padrão
            return self._generate_fallback_response(sentiment, validation_result)
    
    def _generate_fallback_response(self, sentiment: str, validation_result: Dict[str, Any]) -> str:
        """
        Gera resposta padrão quando o LLM falha.
        
        Args:
            sentiment: Sentimento detectado
            validation_result: Resultado da validação
            
        Returns:
            Resposta padrão apropriada
        """
        requer_revisao = validation_result.get('requer_revisao_humana', False)
        
        if requer_revisao:
            return "Obrigado pelo seu feedback! Um membro da nossa equipe irá analisar sua avaliação e entrar em contato em breve para melhor atendê-lo."
        
        if sentiment == "Positivo":
            return "Muito obrigado pelo seu feedback positivo! 😊 Ficamos felizes em saber que você teve uma boa experiência. Conte sempre conosco!"
        elif sentiment == "Negativo":
            return "Lamentamos muito pela sua experiência negativa. 😔 Pedimos sinceras desculpas e vamos trabalhar para resolver isso. Por favor, entre em contato com nosso suporte para que possamos ajudá-lo."
        else:  # Neutro
            return "Obrigado pelo seu feedback! 📝 Valorizamos sua opinião e estamos sempre buscando melhorar. Se tiver alguma sugestão, estamos à disposição!"
