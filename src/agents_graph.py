from .agents.risk_agent import risks_agent
from .tools.rags.risk_rag import search_risks
from .agents.incident_agent import incident_agent
from .tools.rags.incidents_rag import search_incidents
from .services.incidents_reports_etl_service import get_reports_by_ids
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from typing_extensions import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Literal
from pydantic import BaseModel, Field

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

class Risk(BaseModel):
    description: str = Field(description="Description of the risk")
    severity: str = Field(description="Severity of the risk (Low, Medium, High)")

class Action(BaseModel):
    description: str = Field(description="Description of the action taken in the project")
    risks: list[Risk] = Field(description="List of risks associated with this action")

class ProjectAnalysisResult(BaseModel):
    actions: list[Action] = Field(description="List of actions involved in the project and their associated risks")

class RiskAssessment(BaseModel):
    action: str = Field(description="The action associated with the risk")
    risk_description: str = Field(description="Description of the identified risk")
    classification: str = Field(description="Risk classification (e.g., Low, Medium, High, Unknown)")
    analysis_summary: str = Field(description="Detailed summary of the risk analysis based on database matches")
    quick_ref: str | None = Field(description="Quick reference from risk database (QuickRef field)", default=None)
    ev_id: str | None = Field(description="Evidence ID from risk database (Ev_ID field)", default=None)
    risk_category: str | None = Field(description="Risk category from database", default=None)
    risk_subcategory: str | None = Field(description="Risk subcategory from database", default=None)
    entity: str | None = Field(description="Entity involved from database", default=None)
    intent: str | None = Field(description="Intent from database", default=None)
    timing: str | None = Field(description="Timing from database", default=None)
    domain: str | None = Field(description="Domain from database", default=None)
    sub_domain: str | None = Field(description="Sub-domain from database", default=None)

class RiskAssessmentResult(BaseModel):
    assessments: list[RiskAssessment] = Field(description="List of detailed risk assessments")

class IncidentAnalysis(BaseModel):
    action: str = Field(description="The action associated with the incident")
    incident_title: str = Field(description="Title of the relevant incident")
    incident_description: str = Field(description="Description of the incident")
    relevance_explanation: str = Field(description="Explanation of why this incident is relevant to the action")
    reports_ids: list[int] = Field(description="List of report IDs associated with the incident", default_factory=list)

class IncidentAnalysisResult(BaseModel):
    analyses: list[IncidentAnalysis] = Field(description="List of incident analyses")

class AgentState(TypedDict):

    messages: Annotated[list[AnyMessage], operator.add]
    analysis_result: dict # Store the structured result here
    risk_assessments: list[dict] # Store the structured detailed risk assessments here
    incident_analyses: list[dict] # Store structured incident analyses
    project_description: str
    riscos_identificados: list[str]
    classificação_risco: str  # Ex: 'Alto Risco', 'Risco Excessivo'
    contexto_legal: str
    interrupção_pendente: bool
    thread_id: str
    llm_calls: int # Track number of LLM calls


def project_analyst_agent(state: AgentState):
    system_prompt = """You are a Project Analyst Agent responsible for analyzing the proposed project and retrieve details so other agents can make further inspection.
    Your task is to decompose the project into its components and identify all related potential risks.
    You should identify points of interaction with the user, what data is involved in the process, how it will be processed, how the outcome can be used, if any part of that can be invasive or harmful to the user, and any other relevant information that can be useful for a comprehensive risk analysis.
    After analyzing the project, you should return a structured summary that includes:
    1. A list of actions that the project involves, with a brief description of each action.
    2. For each action, a list of potential risks associated with it

    The list of actions should consist of at max 10 actions, the actions shouldn`t be too specific, but rather define the main actions involved in the project.

    Always cite the specific parts of the project description you use in your analysis, so other agents can easily refer to the original context when they need to perform a deeper analysis of any specific risk or action.
    """
    
    structured_llm = llm.with_structured_output(ProjectAnalysisResult)
    result: ProjectAnalysisResult = structured_llm.invoke([SystemMessage(content=system_prompt)] + state["messages"]) #type: ignore
    
    summary = "Project Analysis:\n"
    for action in result.actions:
        summary += f"- Action: {action.description}\n"
        for risk in action.risks:
            summary += f"  - Risk: {risk.description} ({risk.severity})\n"
            
    return {
        "analysis_result": result.model_dump(),
        "messages": [SystemMessage(content=summary)]
    }


def supervisor_agent(state: AgentState) -> AgentState:
    '''Examine the project analysis result and determine which agent(s) should be called for deeper analysis based on the identified risks.
    You should use the risk agent to analyze any identified risks and obtain risk classifications and other details.
    You should use the incident agent to find any relevant past incidents that are similar to the identified risks, to provide more context and insights for the risk analysis.
    After all risks have been analyzed and relevant incidents have been retrieved, you should synthesize all this information into a comprehensive summary that can be presented to the user.'''

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

def risk_agent_call(state: AgentState) -> AgentState:
    analysis_result = state["analysis_result"]
    actions = analysis_result.get("actions", [])
    
    search_results_summary = "Risk Database Search Results:\n"
    
    for action in actions:
        action_desc = action.get("description", "")
        risks = action.get("risks", [])
        
        search_results_summary += f"\nAction: {action_desc}\n"
        
        for risk in risks:
            risk_desc = risk.get("description", "")
            query = f"{risk_desc}"
            
            # Invoke the search tool
            try:
                # search_risks is a tool, invoke with dictionary
                docs = search_risks.invoke({"query": query})
                
                search_results_summary += f"  Identified Potential Risk: {risk_desc}\n"
                search_results_summary += "  Database Matches:\n"
                
                if isinstance(docs, str): # Handle "No risks found" string
                     search_results_summary += f"    - {docs}\n"
                else:
                    for i, doc in enumerate(docs):
                        metadata = doc.metadata
                        search_results_summary += f"    Match {i+1}:\n"
                        search_results_summary += f"      - QuickRef: {metadata.get('quick_ref', 'N/A')}\n"
                        search_results_summary += f"      - Ev_ID: {metadata.get('ev_id', 'N/A')}\n"
                        search_results_summary += f"      - Risk Category: {metadata.get('risk_category', 'N/A')}\n"
                        search_results_summary += f"      - Risk Subcategory: {metadata.get('risk_subcategory', 'N/A')}\n"
                        search_results_summary += f"      - Entity: {metadata.get('entity', 'N/A')}\n"
                        search_results_summary += f"      - Intent: {metadata.get('intent', 'N/A')}\n"
                        search_results_summary += f"      - Timing: {metadata.get('timing', 'N/A')}\n"
                        search_results_summary += f"      - Domain: {metadata.get('domain', 'N/A')}\n"
                        search_results_summary += f"      - Sub-domain: {metadata.get('sub_domain', 'N/A')}\n"
                        search_results_summary += f"      - Description: {doc.page_content[:200]}...\n"

            except Exception as e:
                search_results_summary += f"  Error searching for risk '{risk_desc}': {str(e)}\n"

    system_prompt = """You are an AI Ethics Risk Analysis Agent. 
    Your task is to analyze and summarize AI ethics risks based on the provided search results from a database of AI risks. 
    
    You must output a structured list of risk assessments.
    
    For each action and its associated search results:
    1. Identify the most relevant risk from the database matches.
    2. Extract the metadata fields (QuickRef, Ev_ID, Risk Category, Risk Subcategory, Entity, Intent, Timing, Domain, Sub-domain) from that best match. If no match is found, leave these fields as None.
    3. Provide a 'risk_description' based on the identified potential risk.
    4. Provide a 'classification' (High, Medium, Low, Unknown).
    5. Provide a detailed 'analysis_summary' justifying the classification and explaining the risk.
    """
    
    structured_llm = llm.with_structured_output(RiskAssessmentResult)
    result = structured_llm.invoke([SystemMessage(content=system_prompt), SystemMessage(content=search_results_summary)])
    
    # Create a summary message for the conversation history
    summary_text = "Risk Analysis Completed. Findings:\n"
    for assessment in result.assessments:
        summary_text += f"- Action: {assessment.action}\n"
        summary_text += f"  - Risk: {assessment.risk_description} ({assessment.classification})\n"
        if assessment.quick_ref:
            summary_text += f"  - Ref: {assessment.quick_ref} (Ev_ID: {assessment.ev_id})\n"
        summary_text += f"  - Summary: {assessment.analysis_summary}\n"

    return {
        "messages": state["messages"] + [SystemMessage(content=summary_text)], 
        "risk_assessments": [r.model_dump() for r in result.assessments],
        "llm_calls": state["llm_calls"] + 1
    }

def incident_agent_call(state: AgentState) -> AgentState:
    analysis_result = state["analysis_result"]
    actions = analysis_result.get("actions", [])
    project_description = state.get("project_description", "")
    
    incident_search_summary_list = ["Incident Database Search Results:\n"]
    
    for action in actions:
        action_desc = action.get("description", "")
        
        incident_search_summary_list.append(f"\nAction: {action_desc}\n")
        
        try:
            docs = search_incidents.invoke({
                "project_description": project_description,
                "action": action_desc,
                "top_k": 3
            })
            
            if isinstance(docs, str): # Handle string response (e.g. "No incidents found")
                incident_search_summary_list.append(f"  - {docs}\n")
            else:
                for i, doc in enumerate(docs):
                    metadata = doc.metadata
                    incident_search_summary_list.append(f"    Match {i+1}:\n")
                    incident_search_summary_list.append(f"      - Incident Title: {metadata.get('title', 'N/A')}\n")
                    incident_search_summary_list.append(f"      - Incident Description: {doc.page_content[:300]}...\n")
                    incident_search_summary_list.append(f"      - Deployer: {metadata.get('deployer', 'N/A')}\n")
                    incident_search_summary_list.append(f"      - Harmed Parties: {metadata.get('harmed_parties', 'N/A')}\n")
                    
                    # Extract raw reports IDs
                    reports_meta = metadata.get('reports')
                    parsed_ids = []
                    
                    if reports_meta:
                        import json
                        import ast
                        try:
                            if isinstance(reports_meta, str):
                                try:
                                    parsed_ids = json.loads(reports_meta)
                                except:
                                    parsed_ids = ast.literal_eval(reports_meta)
                            elif isinstance(reports_meta, list):
                                parsed_ids = reports_meta
                        except:
                            pass
                    
                    # Validate IDs are integers/digits
                    valid_ids = [x for x in parsed_ids if isinstance(x, (int, str)) and str(x).isdigit()]
                    if valid_ids:
                        incident_search_summary_list.append(f"      - Related Report IDs: {valid_ids}\n")
                    else:
                        incident_search_summary_list.append("      - Related Report IDs: None\n")

        except Exception as e:
            incident_search_summary_list.append(f"  Error searching for incidents for action '{action_desc}': {str(e)}\n")

    incident_search_summary = "".join(incident_search_summary_list)
    
    system_prompt = """You are an AI Ethics Incident Analysis Agent. 
    Your task is to analyze relevant past AI incidents based on the provided search results from a database.
    
    You must output a structured list of incident analyses.
    
    For each action:
    1. Identify the most relevant past incident if any.
    2. Provide the 'incident_title' and 'incident_description'.
    3. Provide a 'relevance_explanation' describing why this past incident is relevant to the current project action and what risks it highlights.
    4. Extract the 'reports_ids' (list of integers) from the chosen incident's metadata. 
       Do NOT invent IDs. Use exactly the list provided in the search result under 'Related Report IDs'.
    """
    
    structured_llm = llm.with_structured_output(IncidentAnalysisResult)
    result = structured_llm.invoke([SystemMessage(content=system_prompt), SystemMessage(content=incident_search_summary)])
    
    # Enrich the results with actual report data AFTER LLM Analysis
    final_analyses = []
    
    for analysis in result.analyses:
        analysis_dict = analysis.model_dump()
        
        # Initialize reports content field
        analysis_dict['reports'] = []
        
        report_ids = analysis.reports_ids
        if report_ids:
            # Convert 1-based CSV/Report IDs to 0-based data index: index = val - 2
            data_indices = []
            for r_id in report_ids:
                if str(r_id).isdigit():
                    idx = int(r_id) - 2
                    if idx >= 0:
                        data_indices.append(idx)
            
            if data_indices:
                try:
                    fetched_reports = get_reports_by_ids(data_indices)
                    analysis_dict['reports'] = fetched_reports
                except Exception as e:
                    print(f"Error fetching reports for incident '{analysis.incident_title}': {e}")
        
        final_analyses.append(analysis_dict)

    # Create a summary message for the conversation history
    summary_text = "Incident Analysis Completed. Findings:\n"
    for analysis in result.analyses:
        summary_text += f"- Action: {analysis.action}\n"
        summary_text += f"  - Relevant Incident: {analysis.incident_title}\n"
        summary_text += f"  - Relevance: {analysis.relevance_explanation}\n"
        count = len(analysis.reports_ids)
        if count > 0:
            summary_text += f"  - {count} related reports retrieved.\n"

    return {
        "messages": state["messages"] + [SystemMessage(content=summary_text)], 
        "incident_analyses": final_analyses,
        "llm_calls": state["llm_calls"] + 1
    }

def legal_framework_agent_call(state: AgentState) -> AgentState:
    pass

def legal_references_agent_call(state: AgentState) -> AgentState:
    pass

def  final_classifier_agent(state: AgentState) -> AgentState:
    # This agent would take the results from the risk and incident analysis and synthesize a final comprehensive summary for the user.
    pass