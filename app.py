import streamlit as st
import time
from src.data_preprocessing import limpar_texto
from src.prediction_api import prever_sentimento


# --- Configuração do Frontend ---
st.set_page_config(page_title="Análise de Sentimento", layout="wide")

st.title("🧠 Análise de Sentimento de Avaliações")
st.markdown("### Interface de Previsão de Modelos (Naive Bayes & Regressão Logística)")


def exibir_resultado(sentimento):
    """Mapeia o sentimento para a cor e exibe o resultado usando HTML/CSS."""
    color_map = {
        "Positivo": {"bg": "#198754", "text": "#D1E7DD", "emoji": "🟢"},
        "Negativo": {"bg": "#DC3545", "text": "#F8D7DA", "emoji": "🔴"},
        "Neutro": {"bg": "#6C757D", "text": "#E2E3E5", "emoji": "⚫"},
    }

    # Escolhe as cores baseadas no resultado
    colors = color_map.get(sentimento, color_map["Neutro"])
    bg_color = colors["bg"]
    text_color = colors["text"]
    emoji = colors["emoji"]

    # Cria a string HTML formatada
    html_code = f"""
    <div style="background-color: {bg_color}; 
                padding: 15px; 
                border-radius: 5px; 
                color: {text_color}; 
                font-size: 20px; 
                font-weight: bold;">
        {emoji} Sentimento Previsto: {sentimento}
    </div>
    """

    st.markdown(html_code, unsafe_allow_html=True)


with st.container():
    st.subheader("Insira uma avaliação para análise")

    # Entrada 1: Texto
    input_text = st.text_area(
        "Texto da Avaliação:",
        height=150,
        placeholder="Ex: A entrega foi rápida, mas o produto veio com defeito.",
    )

    # Entrada 2: Seleção do Modelo
    model_choice = st.selectbox(
        "Selecione o Modelo para Previsão:",
        options=["Regressão Logística (lr)", "Naive Bayes (nb)"],
    )

    # Botão de Previsão
    if st.button("Analisar Sentimento"):
        if input_text:
            # Extrai a sigla do modelo
            model_key = model_choice.split("(")[-1].replace(")", "")

            # Chama a função de previsão
            with st.spinner(f"Processando com {model_choice}..."):
                start_time = time.time()
                resultado = prever_sentimento(input_text, model_key, limpar_texto)
                end_time = time.time()

                tempo_execucao = end_time - start_time

                if isinstance(resultado, str):
                    st.error(resultado)
                else:
                    # Exibe o resultado principal
                    exibir_resultado(resultado["label"])

                    # --- Seção de Métricas (Requisito do Projeto) ---
                    st.markdown("---")
                    st.subheader("📊 Métricas da Decisão")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("⏱️ Tempo de Execução", f"{tempo_execucao:.4f} s")

                    with col2:
                        confianca = resultado["probabilities"][resultado["label"]]
                        st.metric("🎯 Confiança", f"{confianca:.2%}")

                    st.markdown("#### Probabilidades Detalhadas")
                    probs = resultado["probabilities"]

                    # Exibe barras de progresso para cada classe
                    cols = st.columns(3)
                    for i, (label, prob) in enumerate(probs.items()):
                        with cols[i]:
                            st.markdown(f"**{label}**")
                            st.progress(prob)
                            st.caption(f"{prob:.2%}")
        else:
            st.error("Por favor, insira um texto para analisar.")

# --- Estrutura de rodapé ---
st.sidebar.info("Análise de Sentimentos sob Incerteza")
st.sidebar.markdown("---")
