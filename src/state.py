import json
from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator
from pydantic import BaseModel, Field, field_validator


class Risk(BaseModel):
    description: str = Field(description="Description of the risk")


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
    title: str | None = Field(description="Full title of the primary risk entry from the database", default=None)
    risk_category: str | None = Field(description="Risk category from database", default=None)
    risk_subcategory: str | None = Field(description="Risk subcategory from database", default=None)
    entity: str | None = Field(description="Entity involved from database", default=None)
    intent: str | None = Field(description="Intent from database", default=None)
    timing: str | None = Field(description="Timing from database", default=None)
    domain: str | None = Field(description="Domain from database", default=None)
    sub_domain: str | None = Field(description="Sub-domain from database", default=None)


class RiskAssessmentResult(BaseModel):
    assessments: list[RiskAssessment] = Field(description="List of detailed risk assessments", default_factory=list)


class IncidentAnalysis(BaseModel):
    action: str = Field(description="The action associated with the incident")
    incident_title: str = Field(description="Title of the relevant incident")
    incident_description: str = Field(description="Description of the incident")
    relevance_explanation: str = Field(description="Explanation of why this incident is relevant to the action")
    reports_ids: list[int] = Field(description="List of report IDs associated with the incident", default_factory=list)


class IncidentAnalysisResult(BaseModel):
    analyses: list[IncidentAnalysis] = Field(description="List of incident analyses", default_factory=list)


class FrameworkCompliance(BaseModel):
    aspect: str = Field(description="The aspect of the project being evaluated against the framework")
    framework_reference: str = Field(description="Relevant excerpt or reference from the framework document")
    compliance_status: str = Field(description="Importance level for the project based on the framework: Observação, Precaução, Mitigação, or Crítico")
    explanation: str = Field(description="Explanation of what the framework says about this aspect and what the researcher should consider")
    source_document: str | None = Field(description="Name or title of the source document the excerpt was taken from", default=None)


class FrameworkAnalysisResult(BaseModel):
    compliance_assessments: list[FrameworkCompliance] = Field(description="List of compliance assessments against the framework")


class FinalClassificationResult(BaseModel):
    project_name: str = Field(description="Name of the project")
    risk_level: str = Field(description="Overall risk level (Low, Medium, High, Critical)")
    executive_summary: str = Field(description="Executive summary of the AI ethics evaluation")
    key_recommendations: list[str] = Field(description="List of key recommendations to mitigate risks", default_factory=list)
    identified_risks: list[str] = Field(description="List of all identified risks", default_factory=list)

    @field_validator("identified_risks", "key_recommendations", mode="before")
    @classmethod
    def parse_json_string_list(cls, v):
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
            return [v]
        return v


class QuestionnaireItem(BaseModel):
    statement: str = Field(description="Afirmação em escala de Likert para reflexão ética sobre o projeto de IA")
    options: list[str] = Field(
        description="Opções de resposta em escala de Likert (5 pontos)",
        default_factory=lambda: [
            "Discordo completamente",
            "Discordo",
            "Indeciso",
            "Concordo",
            "Concordo plenamente",
        ],
    )


class QuestionnaireResult(BaseModel):
    title: str = Field(description="Título do questionário de autoavaliação ética")
    description: str = Field(description="Descrição do objetivo do questionário")
    items: list[QuestionnaireItem] = Field(
        description="Lista de 5 a 7 afirmações para autoavaliação ética em escala de Likert"
    )

    @field_validator("items")
    @classmethod
    def validate_item_count(cls, v: list) -> list:
        if len(v) > 7:
            return v[:7]
        return v


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    analysis_result: dict
    risk_assessments: list[dict]
    incident_analyses: list[dict]
    framework_analyses: list[dict]
    project_description: str
    identified_risks: list[str]
    risk_classification: str
    executive_summary: str
    contexto_legal: str
    thread_id: str
    llm_calls: Annotated[int, operator.add]
    questionnaire_items: list[dict]
