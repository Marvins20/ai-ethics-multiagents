from langchain_core.messages import SystemMessage, HumanMessage
from ..state import (AgentState, FinalClassificationResult)
from ..model import model

_LEVEL_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}


def _ceiling_risk(risk_assessments: list[dict]) -> str:
    """Return the highest risk level actually found across individual assessments."""
    max_level = 0
    for r in risk_assessments:
        level = r.get("classification", "Low")
        max_level = max(max_level, _LEVEL_ORDER.get(level, 0))
    return {v: k for k, v in _LEVEL_ORDER.items()}[max_level]


def final_classifier_agent_call(state: AgentState) -> dict:
    project_description = state.get("project_description", "")
    risk_assessments = state.get("risk_assessments", [])
    incident_analyses = state.get("incident_analyses", [])
    framework_analyses = state.get("framework_analyses", [])
    
    context = f"Project Description:\n{project_description}\n\n"
    
    context += "Risk Assessments:\n"
    for r in risk_assessments:
        context += f"- {r.get('risk_description')} ({r.get('classification')}): {r.get('analysis_summary')}\n"
        
    context += "\nIncident Analyses:\n"
    for i in incident_analyses:
        context += f"- {i.get('incident_title')}: {i.get('relevance_explanation')}\n"
        
    context += "\nFramework Compliance Analyses:\n"
    for f in framework_analyses:
        context += f"- {f.get('aspect')} ({f.get('compliance_status')}): {f.get('explanation')}\n"
        
    ceiling = _ceiling_risk(risk_assessments)

    system_prompt = f"""Você é o Auditor-Chefe de Ética em IA.
    Seu papel é sintetizar os achados dos agentes especializados e produzir um resumo final equilibrado.

    IMPORTANTE: Todas as respostas devem estar em português brasileiro.

    REGRA OBRIGATÓRIA DE CLASSIFICAÇÃO: o 'risk_level' final NÃO pode ser superior ao maior nível
    encontrado nas avaliações individuais. O teto calculado com base nas avaliações é: "{ceiling}".
    Portanto, o 'risk_level' deve ser "{ceiling}" ou inferior — NUNCA superior a isso.

    Com base no contexto fornecido:
    1. Defina o 'risk_level' geral (Low, Medium, High, Critical) respeitando o teto acima.
    2. Forneça um 'executive_summary' conciso: máximo 3 frases resumindo o diagnóstico de forma equilibrada.
    3. Liste os 'identified_risks' reais encontrados (frases curtas; omita riscos operacionais ou não éticos).
    4. Forneça 'key_recommendations' acionáveis e proporcionais ao nível de risco real (frases curtas).
    """
    
    structured_llm = model.with_structured_output(FinalClassificationResult)
    result = structured_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=context)])
    
    if isinstance(result, dict):
        result_obj = FinalClassificationResult(**result)
    else:
        result_obj = result

    # Hard cap: final level cannot exceed the ceiling derived from individual assessments
    final_level = result_obj.risk_level  # type: ignore
    if _LEVEL_ORDER.get(final_level, 1) > _LEVEL_ORDER.get(ceiling, 0):
        print(f"[final_classifier] capping {final_level!r} → {ceiling!r}", flush=True)
        final_level = ceiling
    
    # Create the final report message
    report = f"# AI Ethics Audit Report: {result_obj.project_name}\n\n" # type: ignore
    
    report += f"**Overall Risk Level:** {final_level}\n\n"
    
    report += "**Executive Summary:**\n"
    report += f"{result_obj.executive_summary}\n\n" # type: ignore
    
    report += "**Identified Risks:**\n"
    for risk in result_obj.identified_risks: # type: ignore
        report += f"- {risk}\n"
        
    report += "\n**Key Recommendations:**\n"
    for rec in result_obj.key_recommendations: # type: ignore
        report += f"- {rec}\n"
        
    return {
        "messages": [SystemMessage(content=report)],
        "identified_risks": result_obj.identified_risks,  # type: ignore
        "risk_classification": final_level,
        "executive_summary": result_obj.executive_summary,  # type: ignore
        "llm_calls": 1
    }
