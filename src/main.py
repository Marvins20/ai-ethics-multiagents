# from .agents.risk_agent import risks_agent
# from .agents.incident_agent import incident_agent
from dotenv import load_dotenv
load_dotenv()

from langchain.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import operator
import os
from .tools.rags.incidents_rag import search_incidents
from .tools.rags.risk_rag import search_risks

from google import genai

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

tools = [search_incidents, search_risks]

llm_with_tools = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

def should_continue(state: AgentState) -> bool:
    """Check if the last message contains a tool call."""
    results = state["messages"][-1]
    return hasattr(results, 'tool_calls') and len(results.tool_calls) > 0 # type: ignore

system_prompt = """
        You are an AI Ethics Risk Analysis Agent. 
        Your task is to analyze and summarize AI ethics risks based on information retrieved from both database of AI risks and incidents. 
        You will be provided with search results from the database, and your goal is to synthesize this information into a coherent summary that highlights the key details of the risk and correlated incidents, 
        including the nature of the risk, the potential consequences, the parties involved, and any ethical considerations. Use the search results to inform your analysis, and ensure that your summary is clear, concise, and informative. 
        Focus on providing insights into the ethical implications of the risk and any lessons that can be learned from it."
        Please always cite the specific parts of the documents you use in your answers.
    """
 
tools_dict = {tool.name: tool for tool in tools}

def call_llm(state: AgentState) -> AgentState:
    '''Call the LLM with the current state messages and return the new state with updated messages and incremented LLM call count.'''
    new_message = llm_with_tools.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    return {"messages": state["messages"] + [new_message], "llm_calls": state["llm_calls"] + 1}

def retriever_action(state: AgentState) -> AgentState:
    '''Execute tool calls from the LLM's response and return the new state with tool call results added as messages.'''

    tool_calls = state['messages'][-1].tool_calls # type: ignore
    results = []
    for t in tool_calls:
        print(f"Calling tool: {t['name']} with queries: {t['args'].get('query', 'No query provided')}")

        if not t['name'] in tools_dict:
            print(f"Tool {t['name']} not found in tools_dict. Skipping.")
            result  = f"Tool {t['name']} not found. Please Retry and Select a valid tool from the list of available tools."
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result from tool {t['name']}: {result}")
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
    
    print("Tool calls completed. Updating state with results. Back to the model!")
    return {"messages": state["messages"] + results, "llm_calls": state["llm_calls"]}

graph = StateGraph(AgentState)
graph. add_node("llm", call_llm)
graph.add_node("retriever", retriever_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever", False: END}
)
graph.add_edge("retriever", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()

def running_agent():
    print("\n ===RAG AGENT===\n")
    llm_calls = 0

    while True:
        user_input = input("Enter your query (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        messages: list[AnyMessage] = [HumanMessage(content=user_input)]
        
        result = rag_agent.invoke({"messages": messages, "llm_calls": llm_calls})

        print("\n=== Agent Response ===\n")
        print("Result:", result["messages"][-1].content)
        llm_calls = result["llm_calls"]

if __name__ == "__main__":
    running_agent()