
from ..state import (AgentState)
from ..model import model
from langchain_core.messages import ToolMessage


def supervisor_agent(state: AgentState) -> AgentState:
    return state
