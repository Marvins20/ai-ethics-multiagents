"""Lightweight single-node graph for action decomposition only."""
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from ..agents.project_analyst_agent import project_analyst_agent
from ..state import AgentState


def build_decompose_graph():
    wf = StateGraph(AgentState)
    wf.add_node("project_analyst", project_analyst_agent)
    wf.add_edge(START, "project_analyst")
    wf.add_edge("project_analyst", END)
    return wf.compile()
