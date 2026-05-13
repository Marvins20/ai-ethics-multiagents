
from langchain_core.messages import SystemMessage
from ..state import (AgentState, ProjectAnalysisResult)
from ..model import model

system_prompt = """Você é um Agente Analista de Projetos responsável por decompor projetos de pesquisa ou desenvolvimento de IA em suas ações concretas.

    IMPORTANTE: Todas as respostas devem estar em português brasileiro.

    Sua tarefa é mapear TODAS as ações que o pesquisador/equipe realizará ao longo do projeto — sem filtrar por relevância ética.
    A análise ética de cada risco será feita por agentes especializados em etapas posteriores.

    DIRETRIZES DE DECOMPOSIÇÃO:
    • Liste todas as etapas e atividades que o projeto envolve: coleta de dados, desenvolvimento de modelos,
      avaliação, implantação, comunicação de resultados, treinamento de equipes, interação com usuários, etc.
    • Inclua ações metodológicas (revisão de literatura, design de experimentos), técnicas (treinamento de modelo,
      integração de sistemas) e de governança (avaliação por pares, publicação, documentação).
    • Não omita ações por parecerem "sem risco" — a triagem ética é responsabilidade de outro agente.

    Retorne um resumo estruturado contendo:
    1. Uma lista de no máximo 10 ações principais do projeto, com descrição breve e direta.
    2. Para cada ação, de 1 a 3 riscos plausíveis com descrição apenas — sem severidade.
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
            summary += f"  - Risk: {risk.description}\n"

    return {
        "analysis_result": result.model_dump(),
        "messages": [SystemMessage(content=summary)],
        "llm_calls": 1
    }
