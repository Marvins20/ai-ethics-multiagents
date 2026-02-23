from .agents.risk_agent import risks_agent
from .agents.incident_agent import incident_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from typing_extensions import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Literal

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

def risk_agent_call(state:dict):
    """LLM decides whether to call a tool or not"""
    system_prompt = """"
        You are an AI Ethics Risk Analysis Agent. 
        Your task is to analyze and summarize AI ethics risks based on information retrieved from a database of AI risks. 
        You will be provided with search results from the database, and your goal is to synthesize this information into a coherent summary that highlights the key details of the risk, 
        including the nature of the risk, the potential consequences, the parties involved, and any ethical considerations. Use the search results to inform your analysis, and ensure that your summary is clear, concise, and informative. 
        Focus on providing insights into the ethical implications of the risk and any lessons that can be learned from it."
    """

    return {
        "messages": [
            risks_agent.invoke(
                [ SystemMessage(content=system_prompt),] +
                state["messages"]),
        ],
        "llm_calls": state["llm_calls"] + 1
    }

def incident_agent_call(state:dict):
    """LLM decides whether to call a tool or not"""
    system_prompt = """You are an AI Ethics Incident Analysis Agent. 
        Your task is to analyze and summarize AI ethics incidents based on information retrieved from a database of incidents. 
        You will be provided with search results from the database, and your goal is to synthesize this information into a coherent summary that highlights the key details of the incident, 
        including the nature of the incident, the parties involved, 
        the consequences, and any ethical considerations. 
        Use the search results to inform your analysis, and ensure that your summary is clear, concise, and informative. 
        Focus on providing insights into the ethical implications of the incident and any lessons that can be learned from it."""

    return {
        "messages": [
            incident_agent.invoke(
                [ SystemMessage(content=system_prompt),] +
                state["messages"]),
        ],
        "llm_calls": state["llm_calls"] + 1
    }

def supervisor_node(state: MessagesState) -> Literal["risk_agent", "incident_agent"]:
    """
    Classifies the user's intent and routes to the appropriate agent.
    """
    messages = state["messages"]
    system_prompt = """You are a supervisor for an AI Ethics system.
    Your job is to route the user's request to the appropriate agent.
    
    If the user asks about AI risks, you should route to 'risk_agent'.
    If the user asks about AI incidents, you should route to 'incident_agent'.
    If the request is unclear or general, default to 'risk_agent'.
    
    Only output the name of the agent to call.
    """
    
    response = llm.invoke([SystemMessage(content=system_prompt)] + messages)
    content = response.content.lower().strip()
    
    if "incident_agent" in content or "incident" in content:
        return "incident_agent"
    return "risk_agent"

workflow = StateGraph(MessagesState)

workflow.add_node("risk_agent", risk_agent_call)
workflow.add_node("incident_agent", incident_agent_call)

# Add conditional edges from START using supervisor logic
workflow.add_conditional_edges(START, supervisor_node)

workflow.add_edge("risk_agent", END)
workflow.add_edge("incident_agent", END)

graph = workflow.compile()

