"""
Sistema Multi-Agente para Análise de Sentimentos - Interface Web.

Aplicação Streamlit que demonstra a arquitetura de agentes inteligentes PEAS
trabalhando em conjunto para análise completa de avaliações de produtos.

Referência: Russell, S. & Norvig, P. (2020). Artificial Intelligence: A Modern Approach.
"""

import streamlit as st
import time
from src.agents import ManagerAgent

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Sistema Multi-Agente | Análise de Sentimentos",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# ESTILOS CSS CUSTOMIZADOS
# ============================================================================

st.markdown("""
<style>
    /* Reset e configurações globais */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header principal */
    .main-header {
        background: #f8f9fa;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: #1a1a2e;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .main-header h1 {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
        color: #1a1a2e;
    }
    
    .main-header .subtitle {
        font-size: 0.85rem;
        color: #495057;
        font-weight: 400;
        margin-bottom: 0.3rem;
    }
    
    .main-header .reference {
        font-size: 0.7rem;
        color: #6c757d;
        font-style: italic;
        border-top: 1px solid #e9ecef;
        padding-top: 0.4rem;
        margin-top: 0.4rem;
    }
    
    /* Cards de agentes na sidebar */
    .agent-card {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .agent-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    }
    
    .agent-card .agent-name {
        font-weight: 600;
        font-size: 0.95rem;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .agent-card .agent-role {
        font-size: 0.8rem;
        color: #6c757d;
        line-height: 1.4;
    }
    
    .agent-indicator {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }
    
    /* Seção de entrada */
    .input-section {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.04);
    }
    
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #e9ecef;
    }
    
    /* Cards de resultado */
    .result-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border: 1px solid #e9ecef;
        height: 100%;
    }
    
    .result-card-header {
        font-size: 0.85rem;
        font-weight: 500;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.75rem;
    }
    
    .result-card-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    
    .result-card-subtitle {
        font-size: 0.8rem;
        color: #adb5bd;
        margin-top: 0.25rem;
    }
    
    /* Timeline de agentes */
    .agent-timeline {
        position: relative;
        padding-left: 2rem;
    }
    
    .agent-timeline::before {
        content: '';
        position: absolute;
        left: 7px;
        top: 0;
        bottom: 0;
        width: 2px;
        background: #e9ecef;
    }
    
    .timeline-item {
        position: relative;
        padding: 1.25rem 1.5rem;
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        margin-bottom: 1rem;
        margin-left: 1rem;
        transition: all 0.3s ease;
    }
    
    .timeline-item:hover {
        border-color: #dee2e6;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    }
    
    .timeline-item::before {
        content: '';
        position: absolute;
        left: -1.85rem;
        top: 1.5rem;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        border: 3px solid #ffffff;
        box-shadow: 0 0 0 2px currentColor;
    }
    
    .timeline-item .agent-title {
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .timeline-item .agent-summary {
        font-size: 0.9rem;
        color: #495057;
        margin-bottom: 0.75rem;
    }
    
    .timeline-item .agent-details {
        font-size: 0.8rem;
        color: #6c757d;
        background: #f8f9fa;
        padding: 0.75rem;
        border-radius: 8px;
        font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
        white-space: pre-wrap;
        line-height: 1.5;
    }
    
    .time-badge {
        font-size: 0.75rem;
        font-weight: 500;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        background: #f1f3f4;
        color: #5f6368;
    }
    
    /* PEAS display */
    .peas-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.75rem;
    }
    
    .peas-item {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.75rem;
    }
    
    .peas-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #6c757d;
        margin-bottom: 0.25rem;
    }
    
    .peas-value {
        font-size: 0.8rem;
        color: #343a40;
    }
    
    /* Probability bars */
    .prob-container {
        margin: 0.5rem 0;
    }
    
    .prob-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        margin-bottom: 0.25rem;
    }
    
    .prob-bar {
        height: 8px;
        background: #e9ecef;
        border-radius: 4px;
        overflow: hidden;
    }
    
    .prob-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    /* Response section */
    .response-box {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1.5rem;
    }
    
    /* Keywords */
    .keyword-tag {
        display: inline-block;
        background: #e7f1ff;
        color: #0066cc;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.25rem;
        border: 1px solid #cce0ff;
    }
    
    /* Heatmap */
    .heatmap-container {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1.25rem;
        line-height: 2;
        font-size: 1rem;
    }
    
    .heatmap-word {
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        margin: 0 0.1rem;
        transition: all 0.2s ease;
    }
    
    .heatmap-word:hover {
        transform: scale(1.05);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #6c757d;
        font-size: 0.85rem;
        border-top: 1px solid #e9ecef;
        margin-top: 3rem;
    }
    
    /* Cycle visualization */
    .cycle-step {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.8rem;
        background: #f8f9fa;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 500;
        color: #495057;
    }
    
    .cycle-arrow {
        color: #adb5bd;
        font-weight: 400;
    }
    
    /* Stats grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .stat-card {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: #6c757d;
        margin-top: 0.25rem;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Botões primários azul marinho */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        background-color: #1a3a5c !important;
        border-color: #1a3a5c !important;
        color: white !important;
    }
    
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {
        background-color: #0f2a45 !important;
        border-color: #0f2a45 !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INICIALIZAÇÃO
# ============================================================================

# Versão do cache - incrementar para forçar recarregamento após alterações
_CACHE_VERSION = 3

@st.cache_resource
def load_manager(_version=_CACHE_VERSION):
    """Inicializa o agente gerenciador do sistema."""
    return ManagerAgent()

manager = load_manager()

# ============================================================================
# DADOS DOS AGENTES
# ============================================================================

AGENTS_INFO = {
    "SentimentAgent": {
        "name": "Agente de Sentimento",
        "role": "Classificação de polaridade textual via ML",
        "color": "#dc2626",
        "peas": {
            "P": "Acurácia > 90%, F1-Score, Latência < 100ms",
            "E": "Textos em português, vocabulário TF-IDF",
            "A": "Classificar, emitir probabilidades, recusar",
            "S": "Texto bruto, comprimento, OOV rate"
        }
    },
    "ValidationAgent": {
        "name": "Agente de Validação",
        "role": "Arbitragem entre modelos e quantificação de incerteza",
        "color": "#7c3aed",
        "peas": {
            "P": "Escolher melhor modelo, minimizar falsos positivos",
            "E": "Distribuições de probabilidade, comparação NB vs LR",
            "A": "Arbitrar modelos, validar, recomendar revisão",
            "S": "Entropia, spread, confiança, concordância"
        }
    },
    "KeywordAgent": {
        "name": "Agente de Palavras-Chave",
        "role": "Extração de termos relevantes via TF-IDF",
        "color": "#ea580c",
        "peas": {
            "P": "Extrair termos discriminativos",
            "E": "Matriz TF-IDF, vocabulário",
            "A": "Emitir keywords, sinalizar OOV",
            "S": "Scores TF-IDF, termos ausentes"
        }
    },
    "ActionAgent": {
        "name": "Agente de Ação",
        "role": "Decisão tática baseada em regras",
        "color": "#0891b2",
        "peas": {
            "P": "Maximizar satisfação, priorizar críticos",
            "E": "Sentimento, validação, regras de negócio",
            "A": "Recomendar ação, escalar, priorizar",
            "S": "Confiança, status, carga do suporte"
        }
    },
    "ResponseAgent": {
        "name": "Agente de Resposta",
        "role": "Geração de texto via LLM (Gemini 2.0)",
        "color": "#1a3a5c",
        "peas": {
            "P": "Empatia, adequação, tempo < 3s",
            "E": "Contexto, templates, API Gemini",
            "A": "Gerar resposta, usar fallback",
            "S": "Confiança, tópicos sensíveis"
        }
    },
    "ManagerAgent": {
        "name": "Agente Gerenciador",
        "role": "Orquestração e coordenação do pipeline",
        "color": "#4338ca",
        "peas": {
            "P": "Latência total, taxa de sucesso > 95%",
            "E": "Estados dos agentes, fila, recursos",
            "A": "Orquestrar, replanejar, alternar modelos",
            "S": "Tempo de execução, status, falhas"
        }
    }
}

SAMPLE_REVIEWS = [
    "O produto chegou com defeito na bateria e esquenta muito. Quero meu dinheiro de volta.",
    "Produto excelente, superou minhas expectativas! A qualidade é impecável e recomendo.",
    "Comprei há 2 semanas e ainda não chegou. O rastreio não atualiza. Muito decepcionado.",
    "O produto funciona bem, mas achei o preço um pouco alto para o que oferece.",
    "Chegou antes do prazo, muito bem embalado. Está funcionando perfeitamente!"
]

def render_heatmap(text: str, keyword_scores: dict, sentiment: str) -> str:
    """
    Renderiza o texto como heatmap com cores baseadas na relevância das palavras.
    
    Args:
        text: Texto original
        keyword_scores: Dict com {palavra: score_tfidf}
        sentiment: Sentimento predito (positivo/negativo/neutro)
        
    Returns:
        HTML do heatmap
    """
    if not keyword_scores:
        return f"<div class='heatmap-container'>{text}</div>"
    
    # Normalizar scores para range 0-1
    max_score = max(keyword_scores.values()) if keyword_scores else 1
    min_score = min(keyword_scores.values()) if keyword_scores else 0
    score_range = max_score - min_score if max_score != min_score else 1
    
    # Definir cores baseadas no sentimento
    if sentiment == "positivo":
        color_high = "#22c55e"  # Verde
    elif sentiment == "negativo":
        color_high = "#ef4444"  # Vermelho
    else:
        color_high = "#3b82f6"  # Azul
    
    # Processar cada palavra
    words = text.split()
    html_words = []
    
    for word in words:
        # Limpar palavra para comparação
        clean_word = word.lower().strip(".,!?;:\"'()[]{}").strip()
        
        if clean_word in keyword_scores:
            # Calcular intensidade (0 a 1)
            score = keyword_scores[clean_word]
            intensity = (score - min_score) / score_range
            
            # Interpolar opacidade
            opacity = 0.3 + (intensity * 0.7)
            
            html_words.append(
                f"<span class='heatmap-word' style='background-color: {color_high}; "
                f"opacity: {opacity:.2f}; color: #fff; font-weight: 600;' "
                f"title='Score TF-IDF: {score:.4f}'>{word}</span>"
            )
        else:
            html_words.append(f"<span>{word}</span>")
    
    return f"<div class='heatmap-container'>{' '.join(html_words)}</div>"

# ============================================================================
# SIDEBAR - ARQUITETURA DOS AGENTES
# ============================================================================

with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0; border-bottom: 1px solid #e9ecef; margin-bottom: 1.5rem;'>
        <h2 style='font-size: 1.1rem; font-weight: 600; color: #1a1a2e; margin: 0;'>
            Arquitetura Multi-Agente
        </h2>
        <p style='font-size: 0.8rem; color: #6c757d; margin: 0.5rem 0 0 0;'>
            Framework PEAS (Russell & Norvig, 2020)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Ciclo do agente
    st.markdown("""
    <div style='background: #f8f9fa; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem;'>
        <div style='font-size: 0.75rem; font-weight: 600; color: #6c757d; margin-bottom: 0.5rem;'>
            CICLO DE EXECUÇÃO
        </div>
        <div style='display: flex; align-items: center; justify-content: center; gap: 0.4rem; flex-wrap: wrap;'>
            <span class='cycle-step'>Perceive</span>
            <span class='cycle-arrow'>→</span>
            <span class='cycle-step'>Decide</span>
            <span class='cycle-arrow'>→</span>
            <span class='cycle-step'>Act</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Cards dos agentes
    selected_agent = st.selectbox(
        "Selecionar agente para detalhes:",
        list(AGENTS_INFO.keys()),
        format_func=lambda x: AGENTS_INFO[x]["name"]
    )
    
    agent = AGENTS_INFO[selected_agent]
    
    st.markdown(f"""
    <div class='agent-card' style='border-left: 4px solid {agent["color"]};'>
        <div class='agent-name'>
            <span class='agent-indicator' style='background: {agent["color"]};'></span>
            {agent["name"]}
        </div>
        <div class='agent-role'>{agent["role"]}</div>
        <div class='peas-grid' style='margin-top: 1rem;'>
            <div class='peas-item'>
                <div class='peas-label'>Performance</div>
                <div class='peas-value'>{agent["peas"]["P"]}</div>
            </div>
            <div class='peas-item'>
                <div class='peas-label'>Environment</div>
                <div class='peas-value'>{agent["peas"]["E"]}</div>
            </div>
            <div class='peas-item'>
                <div class='peas-label'>Actuators</div>
                <div class='peas-value'>{agent["peas"]["A"]}</div>
            </div>
            <div class='peas-item'>
                <div class='peas-label'>Sensors</div>
                <div class='peas-value'>{agent["peas"]["S"]}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Estatísticas do sistema
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    
    with st.expander("Estatísticas do Sistema", expanded=False):
        stats = manager.get_system_stats()
        st.markdown(f"""
        <div style='font-size: 0.85rem;'>
            <div style='display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #e9ecef;'>
                <span style='color: #6c757d;'>Requisições</span>
                <span style='font-weight: 600;'>{stats['manager']['total_requests']}</span>
            </div>
            <div style='display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #e9ecef;'>
                <span style='color: #6c757d;'>Taxa de Sucesso</span>
                <span style='font-weight: 600;'>{stats['manager']['success_rate']:.1%}</span>
            </div>
            <div style='display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #e9ecef;'>
                <span style='color: #6c757d;'>Replanejamentos</span>
                <span style='font-weight: 600;'>{stats['manager']['replanning_count']}</span>
            </div>
            <div style='display: flex; justify-content: space-between; padding: 0.5rem 0;'>
                <span style='color: #6c757d;'>Agentes Ativos</span>
                <span style='font-weight: 600;'>6</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# CONTEÚDO PRINCIPAL
# ============================================================================

# Header
st.markdown("""
<div class='main-header'>
    <h1>Sistema Multi-Agente para Análise de Sentimentos</h1>
    <p class='subtitle'>
        Arquitetura de agentes inteligentes orientados a objetivos para classificação 
        de sentimentos e geração automática de respostas em avaliações de produtos.
    </p>
    <p class='reference'>
        Fundamentação: Russell, S. & Norvig, P. (2020). Artificial Intelligence: A Modern Approach. 4th ed.
    </p>
</div>
""", unsafe_allow_html=True)

# Seção de entrada
st.markdown("<div class='section-title'>Entrada de Dados</div>", unsafe_allow_html=True)

col_input, col_config = st.columns([3, 1])

with col_input:
    selected_example = st.selectbox(
        "Selecione um exemplo ou digite abaixo:",
        ["Texto personalizado..."] + SAMPLE_REVIEWS,
        label_visibility="collapsed"
    )
    
    default_value = "" if selected_example == "Texto personalizado..." else selected_example
    
    input_text = st.text_area(
        "Avaliação:",
        height=120,
        value=default_value,
        placeholder="Digite ou cole uma avaliação de cliente para análise...",
        label_visibility="collapsed"
    )
    
    if selected_example != "Texto personalizado...":
        input_text = selected_example

with col_config:
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: #e7f3ff; border: 1px solid #b3d9ff; border-radius: 8px; padding: 0.75rem; margin-bottom: 1rem;'>
        <div style='font-size: 0.75rem; font-weight: 600; color: #0066cc; text-transform: uppercase; margin-bottom: 0.25rem;'>Modo Ensemble</div>
        <div style='font-size: 0.8rem; color: #004080;'>Ambos os modelos (NB + LR) são executados e o Agente Validador escolhe o melhor resultado.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sempre usar ensemble
    model_key = "ensemble"
    
    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
    
    analyze = st.button(
        "Iniciar Análise",
        type="primary",
        use_container_width=True
    )

st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

# ============================================================================
# PROCESSAMENTO E RESULTADOS
# ============================================================================

if analyze and input_text:
    # Verificação de entrada
    if len(input_text.split()) < 2:
        st.error("O Agente de Sentimento recusou a requisição: texto muito curto (mínimo 2 palavras).")
        st.info("Esta é uma demonstração da autonomia dos agentes - eles podem recusar requisições inválidas.")
        st.stop()
    
    # Progress indicator
    with st.spinner("Executando pipeline multi-agente..."):
        start_time = time.time()
        res = manager.process(input_text, model_key)
        total_time = time.time() - start_time
    
    # Salvar resultado no session_state
    st.session_state.analysis_result = res
    st.session_state.total_time = total_time
    st.session_state.agent_step = 0  # Resetar navegacao
    
    # Verificar erro
    if "error" in res:
        st.error(f"Erro no processamento: {res.get('error')}")
        st.stop()

# Mostrar resultados se existirem no session_state
if "analysis_result" in st.session_state:
    res = st.session_state.analysis_result
    total_time = st.session_state.get("total_time", 0)
    
    st.markdown("<div class='section-title'>Resultados da Análise</div>", unsafe_allow_html=True)
    
    # Métricas principais
    sent = res["sentiment_analysis"]["label"]
    probs = res["sentiment_analysis"]["probabilities"]
    validation = res.get("validation", {})
    
    color_map = {"Positivo": "#2a9d8f", "Negativo": "#e63946", "Neutro": "#6c757d"}
    sent_color = color_map.get(sent, "#6c757d")
    
    # Verificar se é ensemble para mostrar modelo escolhido
    is_ensemble = True  # Sempre ensemble agora
    chosen_model_display = ""
    if "chosen_model" in res:
        model_names = {"nb": "Naive Bayes", "lr": "Reg. Logística"}
        chosen_model_display = f"via {model_names.get(res['chosen_model'], res['chosen_model'])}"
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        subtitle_text = f"Ensemble {chosen_model_display}" if is_ensemble else "Sentimento detectado"
        st.markdown(f"""
        <div class='result-card' style='border-top: 4px solid {sent_color};'>
            <div class='result-card-header'>Classificação</div>
            <div class='result-card-value' style='color: {sent_color};'>{sent}</div>
            <div class='result-card-subtitle'>{subtitle_text}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        conf = probs[sent]
        st.markdown(f"""
        <div class='result-card'>
            <div class='result-card-header'>Confiança</div>
            <div class='result-card-value'>{conf:.1%}</div>
            <div class='result-card-subtitle'>Probabilidade da classe</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        status = validation.get("status", "N/A")
        status_display = status.replace("_", " ").title()
        st.markdown(f"""
        <div class='result-card'>
            <div class='result-card-header'>Validação</div>
            <div class='result-card-value' style='font-size: 1.4rem;'>{status_display}</div>
            <div class='result-card-subtitle'>Status de confiabilidade</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        exec_time = res.get("total_execution_time_ms", total_time * 1000)
        st.markdown(f"""
        <div class='result-card'>
            <div class='result-card-header'>Tempo Total</div>
            <div class='result-card-value'>{exec_time:.0f}<span style='font-size: 1rem;'>ms</span></div>
            <div class='result-card-subtitle'>Latência do pipeline</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    
    # Seção de Resultados - Layout em 2 colunas
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # Probabilidades e Keywords
        st.markdown("<div class='section-title'>Probabilidades</div>", unsafe_allow_html=True)
        for label, prob in sorted(probs.items(), key=lambda x: -x[1]):
            bar_color = color_map.get(label, "#6c757d")
            st.markdown(f"""
            <div style='margin-bottom: 0.5rem;'>
                <div style='display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 0.2rem;'>
                    <span>{label}</span><span style='font-weight: 600;'>{prob:.0%}</span>
                </div>
                <div style='height: 6px; background: #e9ecef; border-radius: 3px;'>
                    <div style='height: 100%; width: {prob*100}%; background: {bar_color}; border-radius: 3px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        
        # Palavras-chave
        st.markdown("<div class='section-title'>Palavras-Chave</div>", unsafe_allow_html=True)
        keywords = res.get("keywords", [])
        if keywords:
            keywords_html = " ".join([f"<span class='keyword-tag'>{kw}</span>" for kw in keywords[:6]])
            st.markdown(f"<div>{keywords_html}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: #6c757d; font-size: 0.85rem;'>Nenhuma identificada.</p>", unsafe_allow_html=True)
        
        # Heatmap compacto
        keyword_scores = res.get("keyword_scores", {})
        if keyword_scores:
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Mapa de Relevância</div>", unsafe_allow_html=True)
            heatmap_html = render_heatmap(input_text, keyword_scores, sent.lower())
            st.markdown(heatmap_html, unsafe_allow_html=True)
    
    with col_right:
        # Agente de Validação (fusão de Agente Validador + Métricas de Validação)
        st.markdown("<div class='section-title'>Agente de Validação</div>", unsafe_allow_html=True)
        
        entropia = validation.get("entropia", 0)
        entropia_norm = validation.get("entropia_normalizada", 0)
        confianca = validation.get("confianca", 0)
        
        # Seção de comparação de modelos (se disponível)
        if "ensemble_comparison" in res:
            comparison = res["ensemble_comparison"]
            nb_comp = comparison["comparison"]["nb"]
            lr_comp = comparison["comparison"]["lr"]
            chosen = comparison["chosen_model"]
            models_agree = comparison["models_agree"]
            model_names = {"nb": "Naive Bayes", "lr": "Reg. Logística"}
            
            st.markdown(f"""
            <div style='background: #f8f9fa; border-radius: 10px; padding: 1rem; border-left: 4px solid #7c3aed; margin-bottom: 1rem;'>
                <div style='font-size: 0.75rem; color: #6c757d; margin-bottom: 0.25rem;'>Modelo Escolhido</div>
                <div style='font-size: 1.1rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.75rem;'>{model_names[chosen]}</div>
                <div style='font-size: 0.8rem; color: #495057; padding-top: 0.5rem; border-top: 1px solid #e9ecef;'>
                    <div style='margin-bottom: 0.25rem;'>NB: {nb_comp['label']} ({nb_comp['score']:.2f})</div>
                    <div style='margin-bottom: 0.25rem;'>LR: {lr_comp['label']} ({lr_comp['score']:.2f})</div>
                    <div style='margin-top: 0.5rem; font-weight: 600; color: {"#2a9d8f" if models_agree else "#dc3545"};'>
                        {"Modelos concordam" if models_agree else "Modelos divergem"}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Métricas de Incerteza
        st.markdown(f"""
        <div style='background: #f8f9fa; border-radius: 10px; padding: 1rem;'>
            <div style='font-size: 0.75rem; color: #6c757d; margin-bottom: 0.5rem; font-weight: 600;'>Métricas de Incerteza</div>
            <div style='display: flex; justify-content: space-between; margin-bottom: 0.75rem;'>
                <span style='font-size: 0.85rem; color: #6c757d;'>Confiança</span>
                <span style='font-size: 0.85rem; font-weight: 600;'>{confianca:.0%}</span>
            </div>
            <div style='display: flex; justify-content: space-between; margin-bottom: 0.75rem;'>
                <span style='font-size: 0.85rem; color: #6c757d;'>Entropia</span>
                <span style='font-size: 0.85rem; font-weight: 600;'>{entropia:.3f} bits</span>
            </div>
            <div style='display: flex; justify-content: space-between;'>
                <span style='font-size: 0.85rem; color: #6c757d;'>Entropia Norm.</span>
                <span style='font-size: 0.85rem; font-weight: 600;'>{entropia_norm:.0%}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    
    # Seção de resposta
    st.markdown("<div class='section-title'>Ação Recomendada e Resposta Gerada</div>", unsafe_allow_html=True)
    
    col_action, col_response = st.columns([1, 2])
    
    with col_action:
        action = res.get("suggested_action", "Não determinada")
        requires_review = validation.get("requer_revisao_humana", False)
        
        st.markdown(f"""
        <div class='response-box'>
            <div style='font-size: 0.8rem; font-weight: 600; color: #6c757d; text-transform: uppercase; 
                        letter-spacing: 0.5px; margin-bottom: 0.75rem;'>Ação do Sistema</div>
            <div style='font-size: 0.95rem; color: #343a40; line-height: 1.6;'>{action}</div>
            {"<div style='margin-top: 1rem; padding: 0.75rem; background: #fff3cd; border-radius: 8px; font-size: 0.85rem; color: #856404;'>Requer revisao humana</div>" if requires_review else ""}
        </div>
        """, unsafe_allow_html=True)
    
    with col_response:
        response = res.get("generated_reply", "Resposta não gerada.")
        
        st.markdown("""
        <div style='font-size: 0.8rem; font-weight: 600; color: #6c757d; text-transform: uppercase; 
                    letter-spacing: 0.5px; margin-bottom: 0.75rem;'>Resposta Automática (Gemini 2.0 Flash)</div>
        """, unsafe_allow_html=True)
        
        st.text_area(
            "Resposta:",
            value=response,
            height=150,
            label_visibility="collapsed",
            disabled=False
        )
    
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    
    # ============================================================================
    # SECAO DE EXECUCAO DOS AGENTES (PEAS) - NAVEGACAO
    # ============================================================================
    
    st.markdown("<div class='section-title'>Pipeline Multi-Agente</div>", unsafe_allow_html=True)
    
    if "execution_trace" in res and len(res["execution_trace"]) > 0:
        trace = res["execution_trace"]
        total_agents = len(trace)
        
        # Sempre atualizar o trace com os dados mais recentes
        st.session_state.current_trace = trace
        
        # Estado para navegacao
        if "agent_step" not in st.session_state:
            st.session_state.agent_step = 0
        
        # Garantir que o indice esteja dentro dos limites
        if st.session_state.agent_step >= total_agents:
            st.session_state.agent_step = 0
        
        current_idx = st.session_state.agent_step
        step = trace[current_idx]
        
        agent_name = step.get("agent", "Agente")
        exec_time = step.get("execution_time_ms", 0)
        summary = step.get("summary", "")
        details = step.get("details", "")
        
        # Normalizar nome do agente (remover sufixos como "(Árbitro)")
        agent_name_normalized = agent_name.split("(")[0].strip()
        
        # Mapear para PEAS e cor
        agent_key = None
        for key, info in AGENTS_INFO.items():
            if info["name"] == agent_name_normalized or info["name"] in agent_name:
                agent_key = key
                break
        agent_info = AGENTS_INFO.get(agent_key, {}) if agent_key else {}
        peas = agent_info.get("peas", {})
        color = agent_info.get("color", "#6c757d")
        
        # Usar nome normalizado para exibição
        display_name = agent_info.get("name", agent_name_normalized) if agent_info else agent_name_normalized
        
        # Determinar comunicacao
        if current_idx == 0:
            received_from = "Usuario"
            received_data = "Texto da avaliacao"
        else:
            prev = trace[current_idx - 1]
            received_from = prev.get("agent", "").replace("Agente de ", "")
            received_data = prev.get("summary", "")
        
        if current_idx < total_agents - 1:
            sends_to = trace[current_idx + 1].get("agent", "").replace("Agente de ", "")
        else:
            sends_to = "Resultado Final"
        
        # Card do agente usando container nativo
        with st.container():
            # Header
            header_col1, header_col2 = st.columns([3, 1])
            with header_col1:
                st.markdown(f"### {display_name}")
                st.caption(agent_info.get("role", ""))
            with header_col2:
                st.metric("Tempo", f"{exec_time:.1f} ms")
            
            # Resultado
            st.info(f"**Resultado:** {summary}")
            
            # Ciclo PEAS
            st.markdown("**Ciclo PEAS:**")
            peas_col1, peas_col2, peas_col3 = st.columns(3)
            
            with peas_col1:
                st.markdown(f"""
                <div style='background: #e3f2fd; border-radius: 8px; padding: 1rem; text-align: center; height: 100%;'>
                    <div style='font-size: 0.75rem; font-weight: 600; color: #1565c0; text-transform: uppercase;'>PERCEIVE</div>
                    <div style='font-size: 0.85rem; color: #1a1a2e; margin-top: 0.5rem;'>Recebe de {received_from}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with peas_col2:
                st.markdown("""
                <div style='background: #fff8e1; border-radius: 8px; padding: 1rem; text-align: center; height: 100%;'>
                    <div style='font-size: 0.75rem; font-weight: 600; color: #f57c00; text-transform: uppercase;'>DECIDE</div>
                    <div style='font-size: 0.85rem; color: #1a1a2e; margin-top: 0.5rem;'>Processamento</div>
                </div>
                """, unsafe_allow_html=True)
            
            with peas_col3:
                st.markdown(f"""
                <div style='background: #e8eef5; border-radius: 8px; padding: 1rem; text-align: center; height: 100%;'>
                    <div style='font-size: 0.75rem; font-weight: 600; color: #1a3a5c; text-transform: uppercase;'>ACT</div>
                    <div style='font-size: 0.85rem; color: #1a1a2e; margin-top: 0.5rem;'>Envia para {sends_to}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Fluxo de comunicacao
            comm_col1, comm_col2, comm_col3 = st.columns([2, 1, 2])
            with comm_col1:
                st.markdown("**Entrada:**")
                st.caption(received_data[:60] + ("..." if len(received_data) > 60 else ""))
            with comm_col2:
                st.markdown("<div style='text-align: center; font-size: 1.5rem; color: #adb5bd;'>→</div>", unsafe_allow_html=True)
            with comm_col3:
                st.markdown("**Saida:**")
                st.caption(summary[:60] + ("..." if len(summary) > 60 else ""))
        
        # Detalhes tecnicos em expander
        with st.expander("Ver detalhes tecnicos"):
            if details:
                st.code(details, language=None)
            if peas:
                st.markdown("**Especificacao PEAS:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Performance:** {peas.get('P', 'N/A')}")
                    st.markdown(f"**Environment:** {peas.get('E', 'N/A')}")
                with col2:
                    st.markdown(f"**Actuators:** {peas.get('A', 'N/A')}")
                    st.markdown(f"**Sensors:** {peas.get('S', 'N/A')}")
        
        # Fluxo completo - normalizar nomes removendo sufixos como "(Árbitro)"
        agents_names = [t.get("agent", "").split("(")[0].strip().replace("Agente de ", "").replace("Agentes de ", "") for t in trace]
        st.caption(f"Fluxo completo: {' → '.join(agents_names)}")
        
        # Navegacao no rodape
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        
        # Indicador de progresso com pontos
        progress_dots = ""
        for i in range(total_agents):
            if i == current_idx:
                progress_dots += "● "
            elif i < current_idx:
                progress_dots += "○ "
            else:
                progress_dots += "○ "
        st.markdown(f"<div style='text-align: center; color: #1a3a5c; font-size: 1.2rem; letter-spacing: 0.3rem;'>{progress_dots.strip()}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center; color: #6c757d; font-size: 0.8rem; margin-bottom: 1rem;'>Etapa {current_idx + 1} de {total_agents}</div>", unsafe_allow_html=True)
        
        nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
        
        with nav_col1:
            if current_idx > 0:
                if st.button("< Anterior", key="btn_prev", use_container_width=True):
                    st.session_state.agent_step = current_idx - 1
                    st.rerun()
        
        with nav_col2:
            pass  # Espaco vazio no centro
        
        with nav_col3:
            if current_idx < total_agents - 1:
                if st.button("Proximo >", key="btn_next", use_container_width=True, type="primary"):
                    st.session_state.agent_step = current_idx + 1
                    st.rerun()
            else:
                st.button("Fim", key="btn_end", use_container_width=True, disabled=True)
    
    else:
        st.warning("Nenhum trace de execucao disponivel.")

st.markdown("""
<div class='footer'>
    <strong>Trabalho Academico - Introducao a Inteligencia Artificial</strong><br>
    Sistema Multi-Agente com Arquitetura PEAS para Analise de Sentimentos
</div>
""", unsafe_allow_html=True)
