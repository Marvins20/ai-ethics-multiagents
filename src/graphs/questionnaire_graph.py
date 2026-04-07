"""
LangGraph workflow for the Autoavaliação Ética (ethical self-assessment) phase.

This graph runs after the main audit graph. It reads the `identified_risks`
already mapped during the Triagem Ética phase from AgentState and produces a
Likert-scale questionnaire that guides the user through an ethical self-evaluation
of their AI project.
"""
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from ..agents.questionnaire_agent import questionnaire_agent_call
from ..state import AgentState


def build_questionnaire_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Build and compile the questionnaire (Autoavaliação) sub-graph.

    Expects AgentState to already contain `identified_risks`, `risk_classification`,
    and `executive_summary` — populated by the main audit graph.

    Args:
        checkpointer: Optional LangGraph checkpointer for run persistence.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("questionnaire_agent", questionnaire_agent_call)

    workflow.add_edge(START, "questionnaire_agent")
    workflow.add_edge("questionnaire_agent", END)

    return workflow.compile(checkpointer=checkpointer)
