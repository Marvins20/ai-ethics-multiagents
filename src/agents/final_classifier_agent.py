
from langchain_core.messages import SystemMessage, HumanMessage
from ..state import (AgentState, FinalClassificationResult)
from ..model import model


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
        
    system_prompt = """Você é o Auditor-Chefe de Ética em IA.
    Seu papel é sintetizar todos os achados dos agentes especializados (Risco, Incidente, Conformidade) e emitir um veredicto final sobre o projeto.

    IMPORTANTE: Todas as respostas devem estar em português brasileiro.

    Com base no contexto fornecido:
    1. Determine o 'risk_level' geral do projeto (Low, Medium, High, Critical).
    2. Forneça um 'executive_summary' conciso: máximo 3 frases resumindo o diagnóstico.
    3. Liste todos os 'identified_risks' encontrados durante o processo (frases curtas e diretas).
    4. Forneça 'key_recommendations' acionáveis para mitigar os riscos e melhorar a conformidade (frases curtas e diretas).
    """
    
    structured_llm = model.with_structured_output(FinalClassificationResult)
    result = structured_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=context)])
    
    if isinstance(result, dict):
        result_obj = FinalClassificationResult(**result)
    else:
        result_obj = result
    
    # Create the final report message
    report = f"# AI Ethics Audit Report: {result_obj.project_name}\n\n" # type: ignore
    
    report += f"**Overall Risk Level:** {result_obj.risk_level}\n\n" # type: ignore
    
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
        "risk_classification": result_obj.risk_level,  # type: ignore
        "executive_summary": result_obj.executive_summary,  # type: ignore
        "llm_calls": 1
    }
