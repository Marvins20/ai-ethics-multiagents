from ..tools.rags.risk_rag import search_risks
from langchain_core.messages import SystemMessage, HumanMessage
from ..state import (AgentState, RiskAssessmentResult)
from ..model import model

system_prompt = """Você é um Agente de Análise de Riscos Éticos em IA.
    Sua tarefa é analisar riscos éticos com base nos resultados de busca de um banco de dados de riscos de IA.

    IMPORTANTE: Todas as respostas devem estar em português brasileiro.

    Para cada ação e seus resultados de busca:
    1. Identifique os riscos mais relevantes dentre todos os resultados encontrados.
    2. Extraia os campos de metadados da melhor correspondência: QuickRef → quick_ref, Ev_ID → ev_id, Title → title,
       Risk Category, Risk Subcategory, Entity, Intent, Timing, Domain, Sub-domain. Se não houver, deixe como None.
    3. Forneça um 'risk_description' CURTO: máximo 2 frases diretas descrevendo o risco identificado.
    4. Forneça uma 'classification' (High, Medium, Low, Unknown).
    5. Forneça um 'analysis_summary' DETALHADO em português: justifique a classificação, cite as referências pelo quick_ref (ex: Tan2022),
       explique o risco com base nas evidências e descreva possíveis consequências.
    """

_MAX_ACTIONS = 8
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