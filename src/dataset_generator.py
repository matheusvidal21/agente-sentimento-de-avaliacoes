import google.generativeai as genai
import pandas as pd
import json
import time 
import os
from dotenv import load_dotenv


os.makedirs("dataset", exist_ok=True)
DATASET_FILEPATH = "dataset/avaliacoes.csv"

def gerar_dataset():
    """
    Gera um dataset de avaliações de produtos com sentimentos
    positivo, negativo e neutro, usando a API Gemini da Google.
    Salva o dataset gerado em 'dataset/avaliacoes.csv'.'
    """
    if os.path.exists(DATASET_FILEPATH):
        print(f"O arquivo {DATASET_FILEPATH} já existe. Pulando a geração do dataset.")
        return
    
    # --- 1. Configuração da API Key ---
    try:
        load_dotenv()
        GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY').strip()
        
        if not GOOGLE_API_KEY:
            raise ValueError("A variável de ambiente 'GEMINI_API_KEY' não está definida.")
        
        genai.configure(api_key=GOOGLE_API_KEY)
        print("API Key configurada com sucesso!")
    except Exception as e:
        print(f"Erro ao configurar a API Key. Verifique se você criou um arquivo .env com a 'GEMINI_API_KEY'.\n{e}")

    # --- 2. Configuração do Modelo ---
    generation_config = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    }
    model = genai.GenerativeModel(model_name="gemini-2.5-flash",
                                generation_config=generation_config)

    # --- 3. Função de Geração ---
    def gerar_avaliacoes(sentimento, quantidade):
        """Pede ao modelo para gerar avaliações com um sentimento específico."""

        prompt_parts = [
            f"""
            Gere uma lista JSON contendo {quantidade} objetos.
            Cada objeto deve representar uma avaliação de produto (review) em português do Brasil,
            como se fosse para sites como Amazon ou Mercado Livre.

            O sentimento de TODAS as {quantidade} avaliações deve ser: {sentimento}.

            REGRAS DE FORMATAÇÃO:
            1. Retorne APENAS uma string de lista JSON válida. A lista deve conter {quantidade} objetos.
            2. Cada objeto JSON deve ter DUAS chaves: "avaliacao" (o texto da review) e "sentimento" (o rótulo: '{sentimento}').
            3. NÃO inclua "```json" ou "```" no início ou fim da resposta. A resposta deve começar com '[' e terminar com ']'.

            Exemplo de formato de saída (para quantidade=2):
            [
            {{"avaliacao": "Produto muito bom, recomendo.", "sentimento": "{sentimento}"}},
            {{"avaliacao": "Chegou rápido e bem embalado.", "sentimento": "{sentimento}"}}
            ]
            """
        ]

        print(f"Gerando {quantidade} avaliações com sentimento: {sentimento}...")
        try:
            response = model.generate_content(prompt_parts)

            # Limpa a resposta para garantir que seja um JSON válido
            clean_response = response.text.strip().replace("```json", "").replace("```", "").strip()

            # Analisa o JSON
            data = json.loads(clean_response)

            # Verificação extra: o Gemini às vezes retorna um objeto em vez de uma lista
            if isinstance(data, dict):
                print("Aviso: Recebido um objeto único, convertendo para lista.")
                return [data]
            if not isinstance(data, list):
                print(f"Erro: Resposta não foi uma lista JSON. Resposta: {clean_response}")
                return []

            return data

        except Exception as e:
            print(f"Erro ao gerar ou analisar dados para {sentimento}: {e}")
            try:
                # Tenta imprimir a resposta que deu erro
                print(f"Resposta recebida que causou erro: {response.text}")
            except Exception:
                pass # Ignora se 'response' não existir

            print("Aguardando 5 segundos antes de tentar o próximo lote...")
            time.sleep(5)
            return []

    # --- 4. Execução Principal (em lotes) ---
    dataset_list = []
    n_total_por_sentimento = 200
    batch_size = 25 # Pedir 25 de cada vez
    n_batches = n_total_por_sentimento // batch_size # = 8 lotes por sentimento

    sentimentos = ["positivo", "negativo", "neutro"]

    print(f"Iniciando geração de {n_total_por_sentimento*3} avaliações em lotes de {batch_size}...")

    for sentimento in sentimentos:
        print(f"\n--- Processando Sentimento: {sentimento.upper()} ---")
        for i in range(n_batches):
            print(f"Lote {i+1}/{n_batches}...")
            dataset_list.extend(gerar_avaliacoes(sentimento, batch_size))

    print(f"\nTotal de avaliações geradas: {len(dataset_list)}")

    # --- 5. Verificação e Salvamento ---
    if len(dataset_list) > 20:
        # Converter para DataFrame do Pandas
        df_avaliacoes = pd.DataFrame(dataset_list)

        # Salvar em um arquivo CSV
        df_avaliacoes.to_csv(DATASET_FILEPATH, index=False)

        print(f"Dataset salvo com sucesso em '{DATASET_FILEPATH}'")

        # Mostra uma amostra de cada sentimento
        print("\nAmostra dos dados gerados:")
        print(df_avaliacoes['sentimento'].value_counts())
        print("\n")
        print(df_avaliacoes.head(5))
    else:
        print(f"Nenhum dado foi gerado (ou muito pouco foi gerado: {len(dataset_list)}).")
        print("Verifique os logs de erro acima. Pode ser um problema na API Key ou de 'rate limit'.")