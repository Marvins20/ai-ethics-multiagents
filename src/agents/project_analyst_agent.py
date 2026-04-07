
from langchain_core.messages import SystemMessage
from ..state import (AgentState, ProjectAnalysisResult)
from ..model import model

system_prompt = """Você é um Agente Analista de Projetos responsável por decompor projetos de IA e identificar riscos potenciais.

    IMPORTANTE: Todas as respostas devem estar em português brasileiro.

    Sua tarefa é decompor o projeto em suas componentes e identificar todos os riscos relacionados.
    Identifique: pontos de interação com o usuário, quais dados estão envolvidos, como serão processados, como o resultado pode ser usado, se alguma parte pode ser invasiva ou prejudicial, e qualquer outra informação relevante para uma análise de risco abrangente.

    Retorne um resumo estruturado contendo:
    1. Uma lista de ações envolvidas no projeto, com descrição breve e direta de cada uma.
    2. Para cada ação, uma lista de riscos potenciais associados.

    A lista deve ter no máximo 10 ações. As ações não devem ser muito específicas — defina as principais ações envolvidas no projeto.

    Sempre cite as partes específicas da descrição do projeto usadas na análise, para que outros agentes possam se referir ao contexto original.
    """

def project_analyst_agent(state: AgentState):
    # Skip if user already pre-approved an analysis (Phase 2 of two-phase flow)
    if state.get("analysis_result"):
        from langchain_core.messages import SystemMessage as _SM
        return {"messages": [_SM(content="Using pre-approved action analysis.")], "llm_calls": 0}

    structured_llm = model.with_structured_output(ProjectAnalysisResult)
    result: ProjectAnalysisResult = structured_llm.invoke([SystemMessage(content=system_prompt)] + state["messages"]) #type: ignore
    
    summary = "Project Analysis:\n"
    for action in result.actions:
        summary += f"- Action: {action.description}\n"
        for risk in action.risks:
            summary += f"  - Risk: {risk.description} ({risk.severity})\n"
            
    return {
        "analysis_result": result.model_dump(),
        "messages": [SystemMessage(content=summary)],
        "llm_calls": 1
    }