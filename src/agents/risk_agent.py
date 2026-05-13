from ..tools.rags.risk_rag import search_risks
from langchain_core.messages import SystemMessage, HumanMessage
from ..state import (AgentState, RiskAssessmentResult)
from ..model import model

system_prompt = """Você é um Agente de Análise de Riscos ÉTICOS em IA com postura CALIBRADA e PROPORCIONAL.
    Seu objetivo é ajudar pesquisadores a refletir sobre riscos éticos — não alarmá-los nem incluir riscos que não são éticos.
    A triagem ética existe para orientar boas práticas, não para punir ou assustar.

    IMPORTANTE: Todas as respostas devem estar em português brasileiro.

    DEFINIÇÃO DE RISCO ÉTICO — analise APENAS riscos que envolvam:
    • Dano a pessoas ou grupos (discriminação, exclusão, vigilância, manipulação)
    • Violação de privacidade ou uso indevido de dados pessoais
    • Falta de transparência, explicabilidade ou responsabilização (accountability)
    • Viés algorítmico que afete pessoas de forma injusta
    • Impacto negativo sobre autonomia, dignidade ou direitos humanos
    • Concentração de poder, exclusão de stakeholders, falta de representatividade

    NÃO SÃO RISCOS ÉTICOS — se o risco fornecido for deste tipo, classifique como LOW e deixe claro no analysis_summary:
    • Riscos operacionais (ex: baixo engajamento, sobrecarga de avaliadores, atrasos)
    • Riscos de negócio ou viabilidade (ex: adoção insuficiente, custo)
    • Riscos técnicos sem impacto direto sobre pessoas (ex: desempenho, escalabilidade)
    • Riscos de projeto ou gestão (ex: prazo, escopo, recursos)

    CRITÉRIOS DE CLASSIFICAÇÃO — aplique com rigor:

    HIGH — Use apenas quando TODOS forem verdadeiros:
      • O banco de dados retornou correspondência direta e claramente relevante à ação.
      • Há evidência de dano ético real documentado em contextos similares.
      • A probabilidade de ocorrência é alta sem mitigações específicas.
      Exemplos: sistemas autônomos de decisão sobre pessoas, modelos de crédito sem supervisão humana.

    MEDIUM — Use quando:
      • Há correspondência razoável no banco de dados com um risco ético real.
      • O risco depende de como o projeto é implementado e é mitigável com boas práticas.
      Exemplos: coleta de dados de usuários, sistemas de recomendação, modelos preditivos com supervisão.

    LOW — Use quando:
      • O banco de dados não retornou correspondência claramente relevante.
      • O risco é ético mas teórico ou remoto.
      • A ação é metodológica, de revisão, conscientização ou governança.
      • O risco fornecido é operacional/de negócio (não é ético de fato).
      Exemplos: revisões de literatura, avaliações por pares, frameworks de governança, questionários de reflexão.

    UNKNOWN — Use apenas quando não houver dados suficientes.

    REGRAS OBRIGATÓRIAS:
    1. Se o risco descrito não for ético (for operacional, de negócio, técnico ou de gestão), classifique como LOW
       e explique no analysis_summary que este é um risco operacional, não ético, e portanto fora do escopo da triagem.
    2. Se o resultado do banco de dados não corresponder claramente à ação, classifique como LOW.
    3. Ações de governança, revisão, conscientização e metodologia são quase sempre LOW.
    4. Em caso de dúvida entre dois níveis, escolha o MENOR.
    5. A maioria das ações em projetos acadêmicos deve ser LOW ou MEDIUM.

    Para cada ação e seus resultados de busca:
    1. Primeiro avalie: este risco é ético ou operacional/técnico/de negócio?
    2. Extraia os campos de metadados da melhor correspondência: QuickRef → quick_ref, Ev_ID → ev_id, Title → title,
       Risk Category, Risk Subcategory, Entity, Intent, Timing, Domain, Sub-domain. Se não houver, deixe como None.
    3. Forneça um 'risk_description' CURTO e EQUILIBRADO: máximo 2 frases proporcionais ao risco real.
    4. Forneça uma 'classification' calibrada seguindo os critérios acima.
    5. Forneça um 'analysis_summary' DETALHADO e AMIGÁVEL ao usuário:
       - Escreva como se estivesse explicando para o pesquisador, em linguagem clara e acessível.
       - Se houver casos reais relevantes, descreva-os brevemente pelo que aconteceu (ex: "sistemas de contratação automatizada
         que demonstraram favorecer candidatos de determinados grupos..."), sem citar identificadores técnicos como QuickRef,
         Ev_ID, nomes de arquivos ou mencionar que existe um banco de dados ou sistema de busca.
       - Justifique a classificação com base no impacto ético real, não em correspondências técnicas.
       - Se for risco operacional, esclareça que está fora do escopo ético e por isso é classificado como Low.
       - Conclua com 1-2 recomendações práticas e concretas.
    """

_MAX_ACTIONS = 10
_MAX_RISKS_PER_ACTION = 2
_MAX_DOCS_PER_RISK = 2
_DESC_CHARS = 120

def risk_agent_call(state: AgentState) -> AgentState:
    analysis_result = state["analysis_result"]
    actions = analysis_result.get("actions", [])[:_MAX_ACTIONS]

    search_results_summary = "Risk Database Search Results:\n"

    for action in actions:
        action_desc = action.get("description", "")
        risks = action.get("risks", [])[:_MAX_RISKS_PER_ACTION]

        search_results_summary += f"\nAction: {action_desc}\n"

        for risk in risks:
            risk_desc = risk.get("description", "")

            try:
                docs = search_risks.invoke({"query": risk_desc, "top_k": _MAX_DOCS_PER_RISK})

                search_results_summary += f"  Risk: {risk_desc}\n"

                if isinstance(docs, str):
                    search_results_summary += f"    - {docs}\n"
                else:
                    for i, doc in enumerate(docs[:_MAX_DOCS_PER_RISK]):
                        m = doc.metadata
                        search_results_summary += (
                            f"    Match {i+1}: [{m.get('quick_ref','N/A')}] "
                            f"Title: {m.get('title','N/A')} | "
                            f"Ev_ID: {m.get('ev_id','N/A')} | "
                            f"{m.get('risk_category','N/A')} / {m.get('risk_subcategory','N/A')} "
                            f"({m.get('domain','N/A')}) — {doc.page_content[:_DESC_CHARS]}\n"
                        )

            except Exception as e:
                search_results_summary += f"  Error: {e}\n"

    
    structured_llm = model.with_structured_output(RiskAssessmentResult)
    result = structured_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=search_results_summary)])

    summary_text = "Risk Analysis Completed. Findings:\n"
    for assessment in result.assessments: #type: ignore
        summary_text += f"- Action: {assessment.action}\n"
        summary_text += f"  - Risk: {assessment.risk_description} ({assessment.classification})\n"
        if assessment.quick_ref:
            summary_text += f"  - Ref: {assessment.quick_ref} (Ev_ID: {assessment.ev_id})\n"
        summary_text += f"  - Summary: {assessment.analysis_summary}\n"

    return {
        "messages": [SystemMessage(content=summary_text)], 
        "risk_assessments": [r.model_dump() for r in result.assessments], #type: ignore
        "llm_calls": 1 #type: ignore
    }