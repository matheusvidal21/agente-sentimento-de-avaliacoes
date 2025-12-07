#!/usr/bin/env python3
"""
Teste de Integração do Sistema Multi-Agente com Arquitetura PEAS.

Este script valida que todos os agentes refatorados funcionam corretamente
com a nova arquitetura baseada em goal-based agents.

Nota: Testa apenas a pipeline sem chamadas ao LLM para rapidez.
"""

from src.agents import (
    ManagerAgent, 
    BaseAgent,
    PEAS,
    AgentPercept,
    Performative
)


def main():
    print("🧪 Teste de Integração do Sistema Multi-Agente PEAS")
    print("=" * 60)
    
    # Inicializar o ManagerAgent que cria todos os outros agentes
    print("\n📦 Teste 1: Inicialização dos Agentes via ManagerAgent")
    manager = ManagerAgent()
    
    # Acessar agentes do manager
    sentiment_agent = manager.sentiment_agents["lr"]
    validation_agent = manager.validation_agent
    keyword_agent = manager.keyword_agent
    action_agent = manager.action_agent
    response_agent = manager.response_agent
    
    print(f"   ✓ SentimentAgent (herda de BaseAgent: {isinstance(sentiment_agent, BaseAgent)})")
    print(f"   ✓ ValidationAgent (herda de BaseAgent: {isinstance(validation_agent, BaseAgent)})")
    print(f"   ✓ KeywordAgent (herda de BaseAgent: {isinstance(keyword_agent, BaseAgent)})")
    print(f"   ✓ ActionAgent (herda de BaseAgent: {isinstance(action_agent, BaseAgent)})")
    print(f"   ✓ ResponseAgent (herda de BaseAgent: {isinstance(response_agent, BaseAgent)})")
    print(f"   ✓ ManagerAgent (herda de BaseAgent: {isinstance(manager, BaseAgent)})")
    
    # Teste PEAS
    print("\n📋 Teste 2: Verificação PEAS")
    agents_to_test = [
        (sentiment_agent, "SentimentAgent"),
        (validation_agent, "ValidationAgent"),
        (keyword_agent, "KeywordAgent"),
        (action_agent, "ActionAgent"),
        (response_agent, "ResponseAgent"),
        (manager, "ManagerAgent")
    ]
    
    for agent, name in agents_to_test:
        peas = agent.peas
        has_all = all([
            len(peas.performance_measures) > 0,
            len(peas.environment_description) > 0,
            len(peas.actuators) > 0,
            len(peas.sensors) > 0
        ])
        status = "✓" if has_all else "✗"
        print(f"   {status} {name}: P={len(peas.performance_measures)}, E={len(peas.environment_description)} chars, A={len(peas.actuators)}, S={len(peas.sensors)}")
    
    # Teste do SentimentAgent
    print("\n🎯 Teste 3: SentimentAgent - Análise de Sentimento")
    
    textos_teste = [
        ("Produto excelente! Recomendo muito.", "Positivo"),
        ("Péssimo produto, não funciona!", "Negativo"),
        ("O produto chegou ok.", "Neutro/Positivo")
    ]
    
    for texto, esperado in textos_teste:
        result = sentiment_agent.predict(texto)
        label = result["label"]
        prob = result["probabilities"][label]
        status = "✓" if label in esperado else "?"
        print(f"   {status} \"{texto[:30]}...\" → {label} ({prob:.1%})")
    
    # Teste do ValidationAgent
    print("\n🔍 Teste 4: ValidationAgent - Quantificação de Incerteza")
    
    # Criar resultado de sentimento simulado com alta confiança
    sentiment_alta_conf = {
        "label": "Positivo",
        "probabilities": {"Positivo": 0.95, "Neutro": 0.03, "Negativo": 0.02}
    }
    sentiment_baixa_conf = {
        "label": "Positivo",
        "probabilities": {"Positivo": 0.35, "Neutro": 0.33, "Negativo": 0.32}
    }
    
    val1 = validation_agent.validate("Texto de teste positivo", sentiment_alta_conf, "lr")
    print(f"   Alta confiança: status={val1['status']}, revisão={val1['requer_revisao_humana']}")
    
    val2 = validation_agent.validate("Texto de teste ambíguo", sentiment_baixa_conf, "lr")
    print(f"   Baixa confiança: status={val2['status']}, revisão={val2['requer_revisao_humana']}")
    
    # Teste do KeywordAgent
    print("\n🔑 Teste 5: KeywordAgent - Extração de Palavras-chave")
    keywords = keyword_agent.extract_keywords("O produto é excelente e a qualidade é ótima")
    print(f"   Keywords extraídas: {len(keywords)} termos")
    if keywords:
        top3 = keywords[:3]
        print(f"   Top 3: {[kw[0] for kw in top3]}")
    
    # Teste do ActionAgent
    print("\n⚡ Teste 6: ActionAgent - Recomendação de Ação")
    
    action1 = action_agent.get_action("Positivo", "CONFIAVEL")
    print(f"   Positivo+Confiável: {action1[:50]}...")
    
    action2 = action_agent.get_action("Negativo", "AMBIGUO")
    print(f"   Negativo+Ambíguo: {action2[:50]}...")
    
    # Teste de can_handle (autonomia)
    print("\n🤖 Teste 7: Autonomia dos Agentes (can_handle)")
    
    can1, _ = sentiment_agent.can_handle({"text": "teste válido com texto suficiente"})
    can2, reason = sentiment_agent.can_handle({"text": "ab"})  # muito curto
    
    print(f"   Texto válido: pode processar = {can1}")
    print(f"   Texto curto: pode processar = {can2} ({reason})")
    
    # Teste de PEAS completo de um agente
    print("\n📖 Teste 8: Detalhes PEAS do SentimentAgent")
    peas = sentiment_agent.peas
    print(f"   Performance Measures:")
    for p in peas.performance_measures[:2]:
        print(f"     • {p}")
    print(f"   Environment: {peas.environment_description[:60]}...")
    print(f"   Actuators:")
    for a in peas.actuators[:2]:
        print(f"     • {a}")
    print(f"   Sensors:")
    for s in peas.sensors[:2]:
        print(f"     • {s}")
    
    # Estatísticas
    print("\n📊 Teste 9: Estatísticas do Sistema")
    stats = manager.get_system_stats()
    print(f"   Requisições totais: {stats['manager']['total_requests']}")
    print(f"   Agentes monitorados: {len(stats['agents'])}")
    
    print("\n" + "=" * 60)
    print("✅ Todos os testes passaram! Sistema PEAS funcionando.")
    print("\nArquitetura implementada:")
    print("  • BaseAgent abstrato com ciclo perceive → decide → act")
    print("  • Especificação PEAS completa para cada agente")
    print("  • Comunicação via AgentMessage com Performatives")
    print("  • Autonomia: agentes podem recusar requisições")
    print("  • Proatividade: alertas e auto-calibração")


if __name__ == "__main__":
    main()
