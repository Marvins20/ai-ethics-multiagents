"""
LangGraph workflow definition for the full multi-agent audit pipeline.

Separating the graph construction from the CLI runner (main.py) allows the
worker and the FastAPI app to import and use the compiled graph without pulling
in CLI-specific code.
"""
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from ..agents.project_analyst_agent import project_analyst_agent
from ..agents.supervisor_agent import supervisor_agent
from ..agents.risk_agent import risk_agent_call
from ..agents.incident_agent import incident_agent_call
from ..agents.proprietary_framework_agent import proprietary_framework_agent_call
from ..agents.final_classifier_agent import final_classifier_agent_call
from ..state import AgentState


def build_audit_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Build and compile the LangGraph state machine.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. AsyncPostgresSaver).
                      When provided, every graph step is saved so runs can be
                      resumed later using the same thread_id.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("project_analyst", project_analyst_agent)
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("risk_agent", risk_agent_call)
    workflow.add_node("incident_agent", incident_agent_call)
    workflow.add_node("proprietary_framework_agent", proprietary_framework_agent_call)
    workflow.add_node("final_classifier_agent", final_classifier_agent_call)

    workflow.add_edge(START, "project_analyst")
    workflow.add_edge("project_analyst", "supervisor")

    # risk and incident agents in parallel
    workflow.add_edge("supervisor", "risk_agent")
    workflow.add_edge("supervisor", "incident_agent")

    # Both converge at proprietary_framework_agent
    workflow.add_edge("risk_agent", "proprietary_framework_agent")
    workflow.add_edge("incident_agent", "proprietary_framework_agent")

    workflow.add_edge("proprietary_framework_agent", "final_classifier_agent")
    workflow.add_edge("final_classifier_agent", END)

    return workflow.compile(checkpointer=checkpointer)
