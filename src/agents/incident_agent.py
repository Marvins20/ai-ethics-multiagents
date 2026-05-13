import json
from ..tools.rags.incidents_rag import search_incidents
from langchain_core.messages import SystemMessage, HumanMessage
from ..state import AgentState, IncidentAnalysisResult
from ..model import model

system_prompt = """Você é um Agente de Análise de Incidentes Éticos em IA.
    Sua tarefa é analisar incidentes reais pré-selecionados por relevância ao projeto.

    IMPORTANTE: Todas as respostas devem estar em português brasileiro.

    REGRA CRÍTICA DE FILTRAGEM: Inclua na resposta APENAS incidentes que tenham relação
    direta e clara com o domínio, tecnologia ou riscos específicos do projeto descrito.
    Se um incidente não tiver conexão evidente com o projeto (mesmo que apareça na lista),
    OMITA-O completamente da resposta. É preferível retornar poucos incidentes altamente
    relevantes do que muitos incidentes vagamente relacionados.

    Para cada incidente RELEVANTE:
    1. Traduza o 'incident_title' para o português brasileiro.
    2. Forneça um 'incident_description' CURTO: máximo 2 frases descrevendo o que aconteceu.
    3. Forneça um 'relevance_explanation' DETALHADO: explique concretamente por que este
       incidente é relevante para este projeto específico, quais riscos ele evidencia e o
       que pode ser aprendido. Cite elementos do projeto ao explicar a conexão.
    4. Em 'action', descreva qual aspecto ou ação do projeto este incidente exemplifica.
    5. Deixe 'reports_ids' como lista vazia [].
    """

_MAX_ACTIONS = 8
_DOCS_PER_ACTION = 3
_MAX_INCIDENTS = 5
_DESC_CHARS = 150
_DOC_WEIGHTS = [3, 2, 1]
_PROJECT_DOCS = 5
_PROJECT_WEIGHTS = [5, 4, 3, 2, 1]
_MIN_SCORE = 4


def _extract_report_ids(metadata: dict) -> list[int]:
    raw = metadata.get("report_ids")
    if not raw:
        return []
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        return [int(x) for x in parsed if str(x).isdigit()]
    except Exception:
        return []


def _record_hit(
    incident_data: dict[str, dict],
    title: str,
    weight: int,
    report_ids: list[int],
    snippet: str,
) -> None:
    if title not in incident_data:
        incident_data[title] = {"score": 0, "report_ids": report_ids, "snippets": []}
    incident_data[title]["score"] += weight
    incident_data[title]["snippets"].append(snippet)


def incident_agent_call(state: AgentState) -> AgentState:
    analysis_result = state["analysis_result"]
    actions = analysis_result.get("actions", [])[:_MAX_ACTIONS]
    project_description = state.get("project_description", "")

    incident_data: dict[str, dict] = {}

    #  Project and description
    try:
        docs = search_incidents.invoke({
            "project_description": project_description,
            "action": "",
            "top_k": _PROJECT_DOCS,
        })
        if not isinstance(docs, str):
            for i, doc in enumerate(docs[:_PROJECT_DOCS]):
                title = doc.metadata.get("title", "").strip()
                if not title:
                    continue
                weight = _PROJECT_WEIGHTS[i] if i < len(_PROJECT_WEIGHTS) else 1
                _record_hit(
                    incident_data, title, weight,
                    _extract_report_ids(doc.metadata),
                    f"Contexto do projeto: {doc.page_content[:_DESC_CHARS]}",
                )
                print(f"[incident_agent] domain doc[{i}] +{weight}pt title={title!r}", flush=True)
    except Exception as e:
        print(f"[incident_agent] domain search error: {e}", flush=True)

    # Actions
    for idx, action in enumerate(actions):
        action_desc = action.get("description", "")
        try:
            docs = search_incidents.invoke({
                "project_description": project_description,
                "action": action_desc,
                "top_k": _DOCS_PER_ACTION,
            })
            if isinstance(docs, str):
                continue
            for i, doc in enumerate(docs[:_DOCS_PER_ACTION]):
                title = doc.metadata.get("title", "").strip()
                if not title:
                    continue
                weight = _DOC_WEIGHTS[i] if i < len(_DOC_WEIGHTS) else 1
                _record_hit(
                    incident_data, title, weight,
                    _extract_report_ids(doc.metadata),
                    f"Ação: {action_desc[:100]} | {doc.page_content[:_DESC_CHARS]}",
                )
                print(f"[incident_agent] action {idx+1} doc[{i}] +{weight}pt title={title!r}", flush=True)
        except Exception as e:
            print(f"[incident_agent] action {idx+1} error: {e}", flush=True)

    # Filter by minimum score then rank — requires domain hit OR multiple action hits
    filtered = {t: d for t, d in incident_data.items() if d["score"] >= _MIN_SCORE}
    ranked = sorted(filtered.items(), key=lambda x: x[1]["score"], reverse=True)[:_MAX_INCIDENTS]
    print(
        f"[incident_agent] after filter (min={_MIN_SCORE}): {[(t, d['score']) for t, d in ranked]}",
        flush=True,
    )

    # Build prompt for LLM
    lines = [f"Projeto: {project_description[:300]}\n\nIncidentes candidatos ranqueados por relevância:\n"]
    ordered_report_ids: list[list[int]] = []

    for rank, (title, data) in enumerate(ranked):
        ordered_report_ids.append(data["report_ids"])
        lines.append(f"\nIncidente {rank + 1} (pontuação: {data['score']}): {title}\n")
        for snip in data["snippets"][:3]:
            lines.append(f"  - {snip}\n")

    incident_summary = "".join(lines) if ranked else "Nenhum incidente relevante encontrado."

    structured_llm = model.with_structured_output(IncidentAnalysisResult)
    result = structured_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=incident_summary)])

    # Match report_ids by title since LLM may have omitted some incidents
    title_to_ids = {title: data["report_ids"] for title, data in ranked}
    final_analyses = []
    for analysis in result.analyses:  # type: ignore
        analysis_dict = analysis.model_dump()
        title_key = analysis_dict.get("incident_title", "")
        ids = title_to_ids.get(title_key, [])
        analysis_dict["reports_ids"] = ids
        analysis_dict["reports"] = []
        final_analyses.append(analysis_dict)
        print(
            f"[incident_agent] final: title={title_key!r} ids={ids[:3]}",
            flush=True,
        )

    summary_text = f"Incident Analysis Completed. {len(final_analyses)} incidents.\n"
    for a in final_analyses:
        summary_text += f"- {a.get('incident_title')} (ids: {a.get('reports_ids', [])[:3]})\n"

    return {
        "messages": [SystemMessage(content=summary_text)],
        "incident_analyses": final_analyses,
        "llm_calls": 1,  # type: ignore
    }
