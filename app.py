"""
Sistema Multi-Agente para Análise de Sentimentos - Interface Web.

Aplicação Streamlit que demonstra a arquitetura de agentes especializados
trabalhando em conjunto para análise completa de avaliações de produtos.
"""

import streamlit as st
import time
from src.agents import ManagerAgent


st.set_page_config(
    page_title="Análise de Sentimentos", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<div style='margin-bottom: 1.5rem;'>
    <h1 style='margin-bottom: 0.5rem;'>Análise de Sentimentos</h1>
    <p style='font-size: 1rem; color: #666; margin-top: 0;'>
        Arquitetura baseada em agentes autônomos especializados para classificação de sentimentos 
        e geração automática de respostas utilizando NLP e ML.
    </p>
</div>
""", unsafe_allow_html=True)


def load_manager():
    """Inicializa o agente gerenciador do sistema."""
    return ManagerAgent()


manager = load_manager()


def render_heatmap(text: str, explanation: list) -> str:
    """
    Renderiza visualização de contribuição de features com gradiente de cores.
    
    Args:
        text: Texto original
        explanation: Lista de tuplas (palavra, score)
        
    Returns:
        HTML formatado com highlight de palavras
    """
    if not explanation:
        return text
        
    score_map = {word.lower(): score for word, score in explanation}
    words = text.split()
    html_parts = []
    
    for word in words:
        clean_word = "".join(filter(str.isalpha, word)).lower()
        score = score_map.get(clean_word, 0)
        
        bg_color = "transparent"
        if score > 0:
            alpha = min(abs(score) * 2.5, 0.6)
            bg_color = f"rgba(40, 167, 69, {alpha})"
        elif score < 0:
            alpha = min(abs(score) * 2.5, 0.6)
            bg_color = f"rgba(220, 53, 69, {alpha})"
            
        html_parts.append(f'<span style="background-color: {bg_color}; padding: 2px 4px; border-radius: 4px;">{word}</span>')
        
    return " ".join(html_parts)

# Exemplos de avaliações para teste
SAMPLE_REVIEWS = [
    "O produto chegou com defeito na bateria e esquenta muito. Quero meu dinheiro de volta, pessima experiencia.",
    "Produto excelente, superou minhas expectativas! A qualidade é impecável e o preço vale muito a pena. Recomendo!",
    "Comprei há 2 semanas e ainda não chegou. O rastreio não atualiza e ninguém responde no suporte. Muito decepcionado.",
    "O produto funciona bem, mas achei o preço um pouco alto para o que oferece. Esperava mais recursos.",
    "Chegou antes do prazo, muito bem embalado. Estou testando e até agora está funcionando perfeitamente!"
]

# --- Entrada do Usuário ---
with st.container():
    st.markdown("### Entrada de Dados")
    
    col_input, col_config = st.columns([2.5, 1])
    
    with col_input:
        # Dropdown de exemplos primeiro
        selected_example = st.selectbox(
            "Selecione um exemplo ou digite abaixo:",
            ["Digite sua própria avaliação..."] + SAMPLE_REVIEWS,
            help="Selecione um exemplo pré-definido ou digite manualmente"
        )
        
        # Define o valor do text area baseado na seleção
        default_value = "" if selected_example == "Digite sua própria avaliação..." else selected_example
        
        # Text area para entrada
        input_text = st.text_area(
            "Texto de avaliação para análise:", 
            height=140, 
            value=default_value,
            help="Insira uma avaliação de cliente para processamento pela arquitetura multi-agente",
            placeholder="Digite ou cole uma avaliação de cliente aqui...",
            key=f"input_{hash(selected_example)}"  # Key único baseado na seleção
        )
        
        # Se o exemplo for diferente de "Digite sua própria...", forçar o uso do exemplo
        if selected_example != "Digite sua própria avaliação...":
            input_text = selected_example
    
    with col_config:
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        model_opt = st.selectbox(
            "Algoritmo de Classificação", 
            ["Regressão Logística (lr)", "Naive Bayes (nb)"],
            help="Selecione o classificador base treinado"
        )
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        analyze = st.button("Executar Pipeline", type="primary", use_container_width=True)
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        clear_btn = st.button("Limpar", use_container_width=True)
        
        if clear_btn:
            st.rerun()
        
st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)

# --- Processamento e Visualização ---
if analyze and input_text:
    model_key = model_opt.split("(")[-1].replace(")", "")
    
    # Container para timeline animada
    timeline_placeholder = st.empty()
    
    with timeline_placeholder.container():
        st.markdown("### Processando Pipeline Multi-Agente")
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.markdown("Inicializando agentes...")
    
    # Processar
    res = manager.process(input_text, model_key)
    
    # Limpar timeline de processamento
    timeline_placeholder.empty()
    
    st.markdown("---")
    
    # ========== SEÇÃO 1: RESULTADO PRINCIPAL ==========
    st.markdown("### Resultado da Classificação")
    
    sent = res["sentiment_analysis"]["label"]
    color_map = {"Positivo": "#198754", "Negativo": "#DC3545", "Neutro": "#6C757D"}
    color = color_map.get(sent, "#6C757D")
    
    # Layout em 4 colunas para métricas principais
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%); 
                        padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                <div style="font-size: 0.85rem; opacity: 0.9; margin-bottom: 0.3rem;">Sentimento</div>
                <div style="font-size: 1.6rem; font-weight: 700;">{sent}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        conf_percent = res["sentiment_analysis"]["probabilities"][sent]
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #4A5568 0%, #2D3748 100%); 
                        padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                <div style="font-size: 0.85rem; opacity: 0.9; margin-bottom: 0.3rem;">Confiança</div>
                <div style="font-size: 1.6rem; font-weight: 700;">{conf_percent:.1%}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with metric_col3:
        validation = res.get('validation', {})
        status = validation.get('status', 'N/A')
        status_emoji = {
            "CONFIAVEL": "✅",
            "CONFIANCA_MODERADA": "⚠️",
            "BAIXA_CONFIANCA": "❌",
            "AMBIGUO": "🔀",
            "OOD": "⚡"
        }.get(status, "❓")
        
        status_color = {
            "CONFIAVEL": "#198754",
            "CONFIANCA_MODERADA": "#FFA500",
            "BAIXA_CONFIANCA": "#DC3545",
            "AMBIGUO": "#6C757D",
            "OOD": "#E83E8C"
        }.get(status, "#6C757D")
        
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, {status_color} 0%, {status_color}dd 100%); 
                        padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                <div style="font-size: 0.85rem; opacity: 0.9; margin-bottom: 0.3rem;">Validação</div>
                <div style="font-size: 1.1rem; font-weight: 700;">{status_emoji} {status.replace('_', ' ')}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with metric_col4:
        total_time = res.get('total_execution_time_ms', 0)
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #E83E8C 0%, #D63384 100%); 
                        padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                <div style="font-size: 0.85rem; opacity: 0.9; margin-bottom: 0.3rem;">Tempo Total</div>
                <div style="font-size: 1.6rem; font-weight: 700;">{total_time:.0f}ms</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
    
    # ========== SEÇÃO: ANÁLISE DE CONFIABILIDADE ==========
    validation = res.get('validation', {})
    if validation.get('requer_revisao_humana', False):
        st.warning(f"⚠️ **ATENÇÃO:** {validation.get('recomendacao', 'Revisão humana recomendada.')}")
    
    with st.expander("🔍 Ver Análise de Confiabilidade Detalhada", expanded=False):
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            st.markdown("#### Métricas de Incerteza")
            entropia = validation.get('entropia', 0)
            entropia_norm = validation.get('entropia_normalizada', 0)
            st.markdown(f"""
            - **Entropia:** {entropia:.3f} bits ({entropia_norm:.1%} da máxima)
            - **Confiança:** {validation.get('confianca', 0):.1%}
            - **Status:** {validation.get('status', 'N/A')}
            """)
            
            metricas = validation.get('metricas', {})
            if metricas:
                st.markdown(f"""
                - **Tamanho do texto:** {metricas.get('tamanho_texto', 'N/A')} palavras
                - **Spread de probabilidades:** {metricas.get('spread_probabilidades', 0):.3f}
                """)
        
        with col_v2:
            st.markdown("#### Interpretação")
            st.markdown(validation.get('recomendacao', 'N/A'))
            
            if validation.get('requer_revisao_humana', False):
                st.error("🚨 Este caso requer revisão por um especialista humano.")
            else:
                st.success("✅ O sistema pode prosseguir automaticamente.")
    
    st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)
    
    # ========== SEÇÃO 2: DETALHES E AÇÕES ==========
    col_left, col_right = st.columns([1.4, 1])
    
    # COLUNA ESQUERDA: Trace de Execução com Timeline Animada
    with col_left:
        st.markdown("### Timeline de Execução dos Agentes")
        st.caption("Visualização sequencial do processamento multi-agente")
        
        if "execution_trace" in res:
            # Cores para cada tipo de agente
            agent_styles = {
                "Agente de Sentimento": {"color": "#DC3545", "border": "#DC3545"},
                "Agente de Validação": {"color": "#6610F2", "border": "#6610F2"},
                "Agente de Palavras-Chave": {"color": "#FD7E14", "border": "#FD7E14"},
                "Agente de Ação": {"color": "#0DCAF0", "border": "#0DCAF0"},
                "Agente de Resposta": {"color": "#198754", "border": "#198754"}
            }
            
            # Container para animação progressiva
            for idx, step in enumerate(res["execution_trace"], 1):
                agent_name = step['agent']
                style = agent_styles.get(agent_name, {"color": "#6C757D", "border": "#6C757D"})
                exec_time = step.get('execution_time_ms', 0)
                
                # Card do agente com métricas de performance
                st.markdown(f"""
                <div style='background: linear-gradient(90deg, {style['color']}18 0%, {style['color']}05 100%); 
                            border-left: 5px solid {style['border']}; 
                            padding: 1rem 1.2rem; 
                            margin: 0.8rem 0; 
                            border-radius: 0 10px 10px 0;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                        <strong style='color: {style['color']}; font-size: 1.1rem;'>
                            {idx}. {agent_name}
                        </strong>
                        <span style='background: {style['color']}; color: white; 
                                     padding: 0.2rem 0.6rem; border-radius: 12px; 
                                     font-size: 0.8rem; font-weight: 600;'>
                            {exec_time:.1f}ms
                        </span>
                    </div>
                    <div style='color: #555; font-size: 0.95rem; margin-bottom: 0.3rem;'>
                        {step['summary']}
                    </div>
                    <div style='width: 100%; background: #E0E0E0; height: 4px; border-radius: 2px; margin-top: 0.5rem;'>
                        <div style='background: {style['color']}; height: 100%; width: {min(exec_time/10, 100)}%; 
                                    border-radius: 2px; transition: width 0.3s;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Detalhes técnicos em expander
                with st.expander("Ver detalhes técnicos", expanded=False):
                    st.markdown(f"<div style='background-color: #F8F9FA; padding: 0.8rem; border-radius: 6px;'>", unsafe_allow_html=True)
                    st.code(step['details'], language="text")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Visualização de feature importance (apenas para Agente de Sentimento com LR)
                    if step['agent'] == "Agente de Sentimento" and model_key == "lr" and 'explanation' in res:
                        st.markdown("**Feature Importance Visualization:**")
                        html_map = render_heatmap(res['text'], res['explanation'])
                        st.markdown(html_map, unsafe_allow_html=True)
                        st.caption("Intensidade cromática indica peso da feature na classificação")
        else:
            st.warning("Trace de execução não disponível para esta requisição.")
    
    # COLUNA DIREITA: Ações e Respostas
    with col_right:
        # Ação Recomendada
        st.markdown("### Ação Recomendada")
        st.markdown("""
        <div style='background-color: #E8F4FD; padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #0066CC; margin-bottom: 1.5rem;'>
            <strong style='color: #0066CC;'>Sistema baseado em regras</strong>
            <p style='margin: 0.5rem 0 0 0; color: #333;'>{}</p>
        </div>
        """.format(res.get('suggested_action', 'Ação não determinada')), unsafe_allow_html=True)
        
        # Resposta Gerada
        st.markdown("### Resposta Automática")
        st.caption("Gerada por LLM (Gemini 2.5 Flash)")
        
        # Sempre usar a resposta mais recente da análise atual
        current_reply = res.get("generated_reply", "Erro na geração de resposta.")
        
        edited_reply = st.text_area(
            "Resposta editável:", 
            value=current_reply, 
            height=240,
            label_visibility="collapsed",
            key=f"reply_{hash(input_text)}"  # Key único por input
        )
        
        # Botão de enviar com feedback visual
        col_send, col_status = st.columns([1, 2])
        with col_send:
            send_button = st.button("Enviar Resposta", type="secondary", use_container_width=True)
        
        if send_button:
            with col_status:
                with st.spinner("Enviando..."):
                    time.sleep(0.8)  # Simula envio
                st.success("Resposta enviada com sucesso!")
                time.sleep(1.5)

# Rodapé
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: grey; font-size: 0.85em;'>
    <strong>Trabalho Acadêmico - Introdução à Inteligência Artificial</strong><br>
    Arquitetura Multi-Agente para Análise de Sentimentos com NLP
</div>
""", unsafe_allow_html=True)
