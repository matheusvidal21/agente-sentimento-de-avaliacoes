#!/usr/bin/env python3
"""
Script para treinar todos os modelos de análise de sentimento.

Este script executa o pipeline completo:
1. Pré-processamento do dataset (dataset/avaliacoes.csv)
2. Treinamento dos modelos (Naive Bayes, Regressão Logística, K-Means)
3. Salvamento dos modelos em formato .joblib
4. Validação com testes manuais

Modelos gerados em models/:
- nb_modelo_sentimento.joblib (Naive Bayes)
- lr_modelo_sentimento.joblib (Regressão Logística)
- kmeans_perfil.joblib (K-Means, 4 clusters)
- vetorizador_tfidf.joblib (TF-IDF Vectorizer)

Uso:
    python treinar.py
"""

if __name__ == "__main__":
    from src.pipeline import main
    main()
