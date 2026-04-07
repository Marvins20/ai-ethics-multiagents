"""CLI entry point for running the AI ethics audit pipeline interactively."""
from dotenv import load_dotenv
load_dotenv()

import sys
import json
from datetime import datetime

from langchain_core.messages import HumanMessage

from .graphs import build_audit_graph


def running_agent():
    print("\n ===AI ETHICS AGENT===\n")

    if len(sys.argv) > 1:
        user_input = sys.argv[1]
        print(f"Using input from argument: {user_input}")
    else:
        user_input = input("Enter your project description/query: ")

    app = build_audit_graph()
    messages = [HumanMessage(content=user_input)]
    print("Running analysis... this may take a moment.")

    events = app.stream({"messages": messages, "llm_calls": 0}, stream_mode="values")
    final_state = None
    for event in events:
        final_state = event
        if "messages" in event and event["messages"]:
            last_msg = event["messages"][-1]
            print(f"Update from agent: {last_msg.type}")

    result = final_state if final_state else {}
    print("\n=== Analysis Complete ===\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"agent_output_{timestamp}.json"

    serialized_result = result.copy()
    serialized_messages = [
        {"type": msg.type, "content": msg.content}
        for msg in result.get("messages", [])
    ]
    serialized_result["messages"] = serialized_messages

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(serialized_result, f, indent=4, ensure_ascii=False)
    print(f"Result saved to {filename}")

    # Generate readable Markdown report
    md_filename = f"agent_output_{timestamp}.md"
    md_lines = []
    md_lines.append("# Relatório de Análise de Ética em IA")
    md_lines.append(f"**Gerado em:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    project_desc = result.get("project_description", "N/A")
    md_lines.append(f"## Descrição do Projeto\n\n{project_desc}\n")
    risk_class = result.get("risk_classification", "N/A")
    md_lines.append(f"## Classificação Geral de Risco: **{risk_class}**\n")

    risk_assessments = result.get("risk_assessments", [])
    if risk_assessments:
        md_lines.append("## Avaliações de Risco\n")
        for i, r in enumerate(risk_assessments, 1):
            md_lines.append(f"### {i}. {r.get('action', 'N/A')}\n")
            md_lines.append(f"- **Risco:** {r.get('risk_description', 'N/A')}")
            md_lines.append(f"- **Classificação:** {r.get('classification', 'N/A')}")
            md_lines.append(f"- **Categoria:** {r.get('risk_category', 'N/A')} / {r.get('risk_subcategory', 'N/A')}")
            md_lines.append(f"- **Domínio:** {r.get('domain', 'N/A')} / {r.get('sub_domain', 'N/A')}")
            md_lines.append(f"- **Entidade:** {r.get('entity', 'N/A')}")
            md_lines.append(f"- **Intenção:** {r.get('intent', 'N/A')}")
            md_lines.append(f"- **Temporalidade:** {r.get('timing', 'N/A')}")
            md_lines.append(f"\n**Análise:** {r.get('analysis_summary', 'N/A')}\n")

    incident_analyses = result.get("incident_analyses", [])
    if incident_analyses:
        md_lines.append("## Incidentes Relacionados\n")
        for i, inc in enumerate(incident_analyses, 1):
            md_lines.append(f"### {i}. {inc.get('incident_title', 'N/A')}\n")
            md_lines.append(f"- **Ação:** {inc.get('action', 'N/A')}")
            md_lines.append(f"- **Descrição:** {inc.get('incident_description', 'N/A')}")
            md_lines.append(f"- **Relevância:** {inc.get('relevance_explanation', 'N/A')}")
            report_ids = inc.get("reports_ids", [])
            if report_ids:
                md_lines.append(f"- **IDs dos Relatórios:** {', '.join(str(r) for r in report_ids)}")
            md_lines.append("")

    framework_analyses = result.get("framework_analyses", [])
    if framework_analyses:
        md_lines.append("## Conformidade com o Framework\n")
        for i, fw in enumerate(framework_analyses, 1):
            status = fw.get("compliance_status", "N/A")
            status_pt = {"Compliant": "Conforme", "Non-Compliant": "Não Conforme", "Needs Review": "Necessita Revisão"}.get(status, status)
            emoji = {"Compliant": "✅", "Non-Compliant": "❌", "Needs Review": "⚠️"}.get(status, "❓")
            md_lines.append(f"### {i}. {fw.get('aspect', 'N/A')} {emoji}\n")
            md_lines.append(f"- **Status:** {status_pt}")
            md_lines.append(f"- **Referência no Framework:** {fw.get('framework_reference', 'N/A')}")
            md_lines.append(f"\n**Explicação:** {fw.get('explanation', 'N/A')}\n")

    if serialized_messages:
        md_lines.append("## Resultado Final do Agente\n")
        md_lines.append(serialized_messages[-1]["content"])

    md_lines.append(f"\n---\n*Chamadas ao LLM utilizadas: {result.get('llm_calls', 'N/A')}*\n")

    with open(md_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Markdown report saved to {md_filename}")

    if serialized_messages:
        print("\nFinal Output:\n")
        print(serialized_messages[-1]["content"])


if __name__ == "__main__":
    running_agent()
