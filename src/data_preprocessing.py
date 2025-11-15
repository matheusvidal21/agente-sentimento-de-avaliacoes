import pandas as pd
import nltk
import re
import unicodedata
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def limpar_texto(texto):
    '''
    Função para limpar o texto de avaliações:
    - Remove acentos
    - Remove números e caracteres especiais
    - Converte para minúsculas
    '''
    # Converte para string (caso algum número tenha sido gerado)
    texto = str(texto)

    # Remove acentos
    nfkd_form = unicodedata.normalize('NFKD', texto)
    texto_sem_acentos = u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

    # Remove números e caracteres especiais, mantendo só letras e espaços
    # Também converte para minúsculas
    texto_limpo = re.sub(r'[^a-zA-Z\s]', '', texto_sem_acentos, re.I|re.A).lower()

    return texto_limpo


def processar_dados(dataset_filepath):
    """
    Função para carregar, limpar e pré-processar os dados de avaliações de produtos.
    Retorna os conjuntos de treino e teste vetorizados (TF-IDF) e os rótulos correspondentes.
    """
    # --- 1. Downloads de recursos do NLTK ---
    nltk.download('stopwords')
    nltk.download('punkt')

    # --- 2. Carregar o Dataset ---
    try:
        df = pd.read_csv(dataset_filepath)
        print(f"Dataset carregado. Total de {len(df)} avaliações.")
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{dataset_filepath}' não encontrado.")
        exit(1)  
        
    # --- 3. Limpeza e Pré-processamento ---
    df = df.dropna(subset=['avaliacao', 'sentimento']) # Remover linhas que possam ter vindo vazias

    # Mapear os rótulos de texto para números
    df['sentimento_label'] = df['sentimento'].map({
        'negativo': 0,
        'neutro': 1,
        'positivo': 2
    })

    # Verificar se todos os rótulos foram mapeados
    if df['sentimento_label'].isnull().any():
        print("Aviso: Existem rótulos no CSV que não são 'positivo', 'negativo' ou 'neutro'. Verifique o arquivo.")
        df = df.dropna(subset=['sentimento_label']) # Remove linhas com rótulos inesperados

    # Aplica a função de limpeza na coluna 'avaliacao'
    print("Iniciando limpeza do texto...")
    df['avaliacao_limpa'] = df['avaliacao'].apply(limpar_texto)

    print("Texto limpo. Exemplo:")
    print(f"Original: {df['avaliacao'].iloc[0]}")
    print(f"Limpo: {df['avaliacao_limpa'].iloc[0]}")

    # --- 4. Divisão Train/Test Split  ---
    X = df['avaliacao_limpa'] # Features (o texto)
    y = df['sentimento_label']  # Target (o sentimento 0, 1 ou 2)

    '''
    Usamos 25% dos dados para teste e estratificamos por 'y'
    A estratificação garante que a proporção de positivos/negativos/neutros
    seja a mesma no treino e no teste.
    '''
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42, # Para reprodutibilidade
        stratify=y
    )

    print(f"\nDados divididos: {len(y_train)} para treino, {len(y_test)} para teste.")
    print(f"Distribuição de classes no treino:\n{y_train.value_counts(normalize=True)}")

    # --- 5. Vetorização (TF-IDF)  ---
    # Carrega as stopwords em português
    stop_words_pt = stopwords.words('portuguese')

    # Configura o vetorizador
    '''
    Ele vai:
    1. Ignorar stopwords (stop_words=stop_words_pt)
    2. Considerar palavras únicas (unigramas) e pares de palavras (bigramas) (ngram_range=(1, 2))
    3. Ignorar palavras que aparecem em menos de 3 documentos (min_df=3)
    '''
    vectorizer = TfidfVectorizer(
        stop_words=stop_words_pt,
        ngram_range=(1, 2),
        min_df=3
    )

    # TREINA o vetorizador (fit) e TRANSFORMA os dados de treino
    print("Treinando o vetorizador TF-IDF nos dados de treino...")
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # APENAS TRANSFORMA os dados de teste (usando o vocabulário aprendido no treino)
    print("Aplicando o vetorizador nos dados de teste...")
    X_test_tfidf = vectorizer.transform(X_test)

    print("\n--- Processo Concluído! ---")
    print(f"Formato da matriz de treino (observações, features): {X_train_tfidf.shape}")
    print(f"Formato da matriz de teste (observações, features): {X_test_tfidf.shape}")
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer, limpar_texto