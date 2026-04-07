from ..tools.rags.framework_rag import search_framework
from langchain_core.messages import SystemMessage, HumanMessage
from ..state import (AgentState, FrameworkAnalysisResult)
from ..model import model

system_prompt = """Você é um Agente de Conformidade especializado em avaliar projetos contra um framework ético proprietário.
    Sua tarefa é avaliar a conformidade do projeto com base EXCLUSIVAMENTE no contexto do framework fornecido.

    IMPORTANTE: Todas as respostas devem estar em português brasileiro.

    Com base na descrição do projeto, riscos identificados e documentos do framework:
    1. Identifique aspectos-chave do projeto que são regidos pelo framework.
    2. Forneça um 'aspect' CURTO: nome direto do aspecto avaliado (ex: "Transparência algorítmica", "Privacidade dos dados").
    3. Cite a parte específica do framework como 'framework_reference' (trecho literal do documento).
    4. Determine o 'compliance_status' (Compliant, Non-Compliant, Needs Review).
    5. Forneça um 'explanation' DETALHADO em 2 partes: primeiro 1 frase resumindo o veredicto, depois explique o raciocínio conectando o aspecto do projeto ao requisito do framework.

    Se o contexto não contiver informações relevantes para julgar um aspecto, não invente regras.
    """

def proprietary_framework_agent_call(state: AgentState) -> AgentState:
    project_description = state.get("project_description", "")
    risk_assessments = state.get("risk_assessments", [])
    incident_analyses = state.get("incident_analyses", [])

    # Summarize state for search query
    summary_parts = [f"Project Description: {project_description}"]
    
    if risk_assessments:
        risks_summary = ", ".join([r.get('risk_description', '') for r in risk_assessments[:5]]) # Limit to top 5 to avoid too long query
        summary_parts.append(f"Identified Key Risks: {risks_summary}")
        
    if incident_analyses:
        incidents_summary = ", ".join([i.get('incident_title', '') for i in incident_analyses[:3]])
        summary_parts.append(f"Related Incidents: {incidents_summary}")
        
    query_summary = " ".join(summary_parts)
    
    # Translate query to Portuguese before searching the framework (documents are in PT-BR)
    translation_result = model.invoke([
        SystemMessage(content="You are a translator. Translate the following text to Brazilian Portuguese. Output ONLY the translated text, nothing else."),
        HumanMessage(content=query_summary)
    ])
    query_in_portuguese = translation_result.content

    # search_framework uses a semantic search, so a robust paragraph works well
    framework_docs = search_framework.invoke({"query": query_in_portuguese})
    
    framework_context = ""
    if isinstance(framework_docs, str):
        framework_context = framework_docs
    else:
        for i, doc in enumerate(framework_docs):
            framework_context += f"Document {i+1} (Source: {doc.metadata.get('source', 'Unknown')}):\n{doc.page_content}\n\n"

    structured_llm = model.with_structured_output(FrameworkAnalysisResult)
    # create a formatted user message with all context
    user_message = f"""
    Project Context:
    {query_summary}
    
    Proprietary Framework Context Retrieved:
    {framework_context}
    """
    
    result = structured_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_message)])
    
    summary_text = "Proprietary Framework Analysis Completed. Findings:\n"
    for assessment in result.compliance_assessments: #type: ignore 
        summary_text += f"- Aspect: {assessment.aspect}\n"
        summary_text += f"  - Status: {assessment.compliance_status}\n"
        summary_text += f"  - Reference: {assessment.framework_reference}\n"
        summary_text += f"  - Explanation: {assessment.explanation}\n"

    return {
        "messages": [SystemMessage(content=summary_text)],
        "framework_analyses": [c.model_dump() for c in result.compliance_assessments], #type: ignore
        "llm_calls": 2 #type: ignore
    } 