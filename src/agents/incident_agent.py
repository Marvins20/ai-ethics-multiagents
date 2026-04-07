import json
from ..tools.rags.incidents_rag import search_incidents
from langchain_core.messages import SystemMessage, HumanMessage
from ..state import AgentState, IncidentAnalysisResult
from ..model import model

system_prompt = """Você é um Agente de Análise de Incidentes Éticos em IA.
    Sua tarefa é analisar incidentes reais já ranqueados por relevância para o projeto.

    IMPORTANTE: Todas as respostas devem estar em português brasileiro.

    Para cada incidente listado (já ordenados por pontuação de relevância):
    1. Mantenha o 'incident_title' exatamente como fornecido.
    2. Forneça um 'incident_description' CURTO: máximo 2 frases descrevendo o que aconteceu.
    3. Forneça um 'relevance_explanation' DETALHADO: explique por que este incidente é relevante para o projeto, quais riscos ele evidencia e o que pode ser aprendido.
    4. Em 'action', copie o contexto da ação mais relevante mostrado para este incidente (após "Ação:").
    5. Deixe 'reports_ids' como lista vazia [].
    6. Produza EXATAMENTE um item por incidente, na mesma ordem apresentada.
    """

_MAX_ACTIONS = 8
_DOCS_PER_ACTION = 3
_MAX_INCIDENTS = 8
_DESC_CHARS = 150
_DOC_WEIGHTS = [3, 2, 1]


def _extract_report_ids(metadata: dict) -> list[int]:
    """Read raw report_number IDs stored in Chroma metadata during ingestion."""
    raw = metadata.get("report_ids")
    if not raw:
        return []
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        return [int(x) for x in parsed if str(x).isdigit()]
    except Exception:
        return []


def incident_agent_call(state: AgentState) -> AgentState:
    analysis_result = state["analysis_result"]
    actions = analysis_result.get("actions", [])[:_MAX_ACTIONS]
    project_description = state.get("project_description", "")

    # Score unique incidents across all actions.
    # title → {score, report_ids, snippets}
    incident_data: dict[str, dict] = {}

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
                if title not in incident_data:
                    incident_data[title] = {
                        "score": 0,
                        "report_ids": _extract_report_ids(doc.metadata),
                        "snippets": [],
                    }
                incident_data[title]["score"] += weight
                snippet = f"Ação: {action_desc[:100]} | {doc.page_content[:_DESC_CHARS]}"
                incident_data[title]["snippets"].append(snippet)
                print(
                    f"[incident_agent] action {idx+1} doc[{i}] +{weight}pt title={title!r}",
                    flush=True,
                )
        except Exception as e:
            print(f"[incident_agent] action {idx+1} error: {e}", flush=True)

    # Sort by relevance score descending, take top _MAX_INCIDENTS
    ranked = sorted(incident_data.items(), key=lambda x: x[1]["score"], reverse=True)[:_MAX_INCIDENTS]
    print(f"[incident_agent] ranked incidents: {[(t, d['score']) for t, d in ranked]}", flush=True)

    # Build prompt for LLM
    lines = ["Incidentes ranqueados por relevância para o projeto:\n"]
    ordered_report_ids: list[list[int]] = []

    for rank, (title, data) in enumerate(ranked):
        ordered_report_ids.append(data["report_ids"])
        lines.append(f"\nIncidente {rank + 1} (pontuação: {data['score']}): {title}\n")
        for snip in data["snippets"][:3]:
            lines.append(f"  - {snip}\n")

    incident_summary = "".join(lines) if ranked else "Nenhum incidente encontrado."

    structured_llm = model.with_structured_output(IncidentAnalysisResult)
    result = structured_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=incident_summary)])

    # Attach report_ids positionally (LLM output order matches input order)
    final_analyses = []
    for i, analysis in enumerate(result.analyses):  # type: ignore
        analysis_dict = analysis.model_dump()
        ids = ordered_report_ids[i] if i < len(ordered_report_ids) else []
        analysis_dict["reports_ids"] = ids
        analysis_dict["reports"] = []
        final_analyses.append(analysis_dict)
        print(
            f"[incident_agent] final[{i}]: title={analysis_dict.get('incident_title')!r} "
            f"ids={ids[:3]}",
            flush=True,
        )

    summary_text = f"Incident Analysis Completed. {len(final_analyses)} unique incidents ranked by relevance.\n"
    for a in final_analyses:
        summary_text += f"- {a.get('incident_title')} (ids: {a.get('reports_ids', [])[:3]})\n"

    return {
        "messages": [SystemMessage(content=summary_text)],
        "incident_analyses": final_analyses,
        "llm_calls": 1,  # type: ignore
    }
