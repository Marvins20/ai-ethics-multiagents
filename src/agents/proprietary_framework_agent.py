import os
from ..tools.rags.framework_rag import search_framework
from langchain_core.messages import SystemMessage, HumanMessage
from ..state import (AgentState, FrameworkAnalysisResult)
from ..model import model

system_prompt = """Você é um Agente de Análise de Frameworks de Referência em Ética de IA.
    Sua tarefa é interpretar o que documentos de referência dizem sobre aspectos do projeto e classificar
    a importância de cada ponto para o pesquisador, com base no que o próprio documento indica.

    IMPORTANTE: Todas as respostas devem estar em português brasileiro.

    POSTURA: Trate o framework como uma referência de consulta, não como lei obrigatória.
    O objetivo é informar o pesquisador sobre o que documentos relevantes recomendam ou alertam,
    para que ele possa tomar decisões mais conscientes.

    Com base na descrição do projeto e nos trechos do documento de referência:
    1. Identifique aspectos do projeto que o documento aborda ou que são relevantes para ele.
    2. Forneça um 'aspect' CURTO: nome do aspecto ético ou técnico em questão (ex: "Transparência algorítmica").
    3. Em 'framework_reference': cite um trecho representativo do documento que justifica a análise.
       Não mencione que veio de um banco de dados ou sistema de busca.
    4. Em 'compliance_status': classifique a IMPORTÂNCIA deste aspecto para o projeto com base no que o
       documento diz. Use EXATAMENTE um dos seguintes valores:
       - "Observação"  → O documento menciona o aspecto como algo de baixo impacto ou baixa probabilidade.
                         Requer apenas monitoramento ocasional. Use quando o documento recomenda atenção leve.
       - "Precaução"   → O documento indica risco moderado que pode causar atrasos ou problemas leves.
                         O pesquisador deve ter um plano de contingência simples.
       - "Mitigação"   → O documento aponta alto potencial de dano. A diretriz é evitar a situação ou criar
                         barreiras. Use quando o documento desaconselha fortemente ou alerta para riscos sérios.
       - "Crítico"     → O documento trata o aspecto como proibido, ilegal, criminoso, perigoso ou de risco
                         alto/urgente. Exige atenção imediata. Use quando o documento usa linguagem de proibição,
                         obrigatoriedade legal, risco grave ou impacto que pode paralisar o projeto.
       Baseie sua escolha no que o documento diz, não numa avaliação geral do projeto.
    5. Em 'explanation': explique de forma AMIGÁVEL e DIRETA o que o documento diz sobre esse aspecto
       e o que o pesquisador deveria considerar. Escreva como orientação útil em 2-3 frases.
       Prefira "o documento recomenda", "vale atenção para", "segundo as diretrizes consultadas".
    6. Em 'source_document': indique o nome do documento de origem consultado exatamente como aparece
       no campo "Fonte:" de cada trecho do contexto (ex: "EU AI Act", "ISO 42001").
       Este campo será exibido ao usuário como referência bibliográfica.

    Se o documento não tiver informações relevantes para um aspecto, não force uma análise.
    Gere apenas itens onde o documento oferece orientação genuína.
    """


def proprietary_framework_agent_call(state: AgentState) -> AgentState:
    project_description = state.get("project_description", "")
    risk_assessments = state.get("risk_assessments", [])
    incident_analyses = state.get("incident_analyses", [])

    summary_parts = [f"Project Description: {project_description}"]
    if risk_assessments:
        risks_summary = ", ".join([r.get("risk_description", "") for r in risk_assessments[:5]])
        summary_parts.append(f"Identified Key Risks: {risks_summary}")
    if incident_analyses:
        incidents_summary = ", ".join([i.get("incident_title", "") for i in incident_analyses[:3]])
        summary_parts.append(f"Related Incidents: {incidents_summary}")

    query_summary = " ".join(summary_parts)

    translation_result = model.invoke([
        SystemMessage(content="You are a translator. Translate the following text to Brazilian Portuguese. Output ONLY the translated text, nothing else."),
        HumanMessage(content=query_summary),
    ])
    query_in_portuguese = translation_result.content

    framework_docs = search_framework.invoke({"query": query_in_portuguese})

    framework_context = ""
    if isinstance(framework_docs, str):
        framework_context = framework_docs
    else:
        for i, doc in enumerate(framework_docs):
            raw_source = doc.metadata.get("source", "Unknown")
            source_name = os.path.splitext(os.path.basename(raw_source))[0].replace("_", " ")
            framework_context += f"[Trecho {i+1} — Fonte: {source_name}]\n{doc.page_content}\n\n"

    structured_llm = model.with_structured_output(FrameworkAnalysisResult)
    user_message = f"""
    Project Context:
    {query_summary}

    Proprietary Framework Context Retrieved:
    {framework_context}
    """

    result = structured_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_message)])

    assessments_return = [compl.model_dump() for compl in result.compliance_assessments]  # type: ignore

    summary_text = "Proprietary Framework Analysis Completed. Findings:\n"
    for d in assessments_return:
        summary_text += f"- {d['aspect']}: {d['explanation']}\n"

    return {
        "messages": [SystemMessage(content=summary_text)],
        "framework_analyses": assessments_return,
        "llm_calls": 2,  # type: ignore
    }
