"""
Módulo de Pré-processamento de Dados.

Responsável pela limpeza textual, vetorização TF-IDF e preparação
dos dados para treinamento dos modelos de análise de sentimento.
"""

import pandas as pd
import nltk
import re
import unicodedata
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def limpar_texto(texto: str) -> str:
    """
    Remove acentos, caracteres especiais e normaliza texto.
    
    Args:
        texto: Texto bruto da avaliação
        
    Returns:
        Texto limpo (sem acentos, apenas letras, minúsculas)
    """
    texto = str(texto)
    
    nfkd_form = unicodedata.normalize('NFKD', texto)
    texto_sem_acentos = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    texto_limpo = re.sub(r'[^a-zA-Z\s]', '', texto_sem_acentos, re.I|re.A).lower()

    return texto_limpo


def processar_dados(dataset_filepath: str) -> tuple:
    """
    Pipeline completo de pré-processamento.
    
    Carrega dataset, limpa textos, mapeia rótulos, aplica vetorização
    TF-IDF e divide em treino/teste com estratificação.
    
    Args:
        dataset_filepath: Caminho para o arquivo CSV do dataset
        
    Returns:
        Tupla (X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer, limpar_texto)
    """
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

    try:
        df = pd.read_csv(dataset_filepath)
        print(f"Dataset carregado. Total de {len(df)} avaliações.")
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{dataset_filepath}' não encontrado.")
        exit(1)  
        
    df = df.dropna(subset=['avaliacao', 'sentimento'])

    df['sentimento_label'] = df['sentimento'].map({
        'negativo': 0,
        'neutro': 1,
        'positivo': 2
    })

    if df['sentimento_label'].isnull().any():
        print("Aviso: Existem rótulos inválidos no CSV. Removendo linhas...")
        df = df.dropna(subset=['sentimento_label'])

    print("Iniciando limpeza do texto...")
    df['avaliacao_limpa'] = df['avaliacao'].apply(limpar_texto)

    print("Texto limpo. Exemplo:")
    print(f"Original: {df['avaliacao'].iloc[0]}")
    print(f"Limpo: {df['avaliacao_limpa'].iloc[0]}")

    X = df['avaliacao_limpa']
    y = df['sentimento_label']

    # Split estratificado: mantém proporção de classes em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    print(f"\nDados divididos: {len(y_train)} para treino, {len(y_test)} para teste.")
    print(f"Distribuição de classes no treino:\n{y_train.value_counts(normalize=True)}")

    stop_words_pt = stopwords.words('portuguese')

    # TF-IDF com unigramas e bigramas
    vectorizer = TfidfVectorizer(
        stop_words=stop_words_pt,
        ngram_range=(1, 2),
        min_df=3
    )

    print("Treinando o vetorizador TF-IDF nos dados de treino...")
    X_train_tfidf = vectorizer.fit_transform(X_train)

    print("Aplicando o vetorizador nos dados de teste...")
    X_test_tfidf = vectorizer.transform(X_test)

    print("\n--- Processo Concluído! ---")
    print(f"Formato da matriz de treino: {X_train_tfidf.shape}")
    print(f"Formato da matriz de teste: {X_test_tfidf.shape}")
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer, limpar_texto