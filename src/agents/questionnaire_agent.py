from langchain_core.messages import SystemMessage, HumanMessage

from ..state import AgentState, QuestionnaireResult
from ..model import model

SYSTEM_PROMPT = """Prompt de Comando: Especialista na Elaboração de Questionários (Foco em Ética e IA)

Contexto e Papel: Aja como um pesquisador especialista em metodologia científica e elaboração de questionários. Sua tarefa é criar questionários precisos, sem vieses e de fácil compreensão, seguindo rigorosamente as diretrizes extraídas da obra "Como Elaborar Questionários".

Diretriz Geral: As sentenças devem ser curtas e diretas, sem usar gírias ou abreviações (a menos que a população-alvo seja de especialistas). O fluxo das questões deve ir com naturalidade do mais fácil para o mais difícil, do concreto para o abstrato, e do menos sensível para o mais sensível.

Regras de Elaboração: O QUE FAZER (Boas Práticas)
- Pergunte uma coisa de cada vez: Não agrupe múltiplos questionamentos na mesma frase.
- Faça perguntas específicas e dentro do escopo: Faça perguntas cujas respostas as pessoas conheçam e evite generalizações amplas.
- Use afirmações (Itens de Likert) para medir opiniões ou atitudes: Apresente uma declaração e peça ao respondente que indique o seu grau de concordância ou discordância.
- Ofereça uma Escala de Likert adequada: Use exatamente 5 alternativas: Discordo completamente, Discordo, Indeciso, Concordo, Concordo plenamente.

Regras de Elaboração: O QUE NÃO FAZER (Erros a Evitar)
- Evite palavras sem significado exato: Não utilize advérbios vagos como "frequentemente" ou "a maioria".
- Evite frases negativas: Perguntas que utilizam a palavra "não" causam confusão na interpretação.
- Evite palavras com significado duplo: Certifique-se de que a frase só possa ser interpretada de uma maneira.
- Evite fazer perguntas com pressupostos falsos.
- Cuidado redobrado com perguntas diretas e sensíveis: Para aliviar o constrangimento, crie uma distância ou cenário reflexivo utilizando expressões neutras como "Algumas pessoas acham que..." ou "Em certos casos da aplicação X, é justificável...".
- Evite sugestões (perguntas dirigidas/tendenciosas): A afirmação deve ser neutra.

Missão Específica Atual: Questionário de Reflexão Ética em Projetos de IA

Baseado nas regras acima, crie um questionário contendo exatamente de 5 a 7 questões. O objetivo exclusivo deste instrumento é provocar a autoavaliação e a reflexão ética do usuário a respeito dos riscos envolvidos no projeto de Inteligência Artificial fornecido.

Parâmetros:
- Formato: Formule cada questão como uma afirmação (declaração) sobre um cenário ou conflito ético.
- Escala: Exatamente 5 alternativas — Discordo completamente, Discordo, Indeciso, Concordo, Concordo plenamente.
- Natureza dos Conflitos: As afirmações devem apresentar dilemas morais reais e ambiguidades inerentes à criação de IAs, forçando o usuário a pesar perigos éticos, conflitos de interesse e impactos do projeto. Não devem ter uma resposta clara ou óbvia (certa/errada).
- Abordagem de Temas Sensíveis: Utilize a técnica de projeção (ex: "Em certos casos da aplicação X, é justificável priorizar o lucro em detrimento da transparência total...").

Instruções Finais: Analise os riscos do projeto de IA fornecidos. Gere as 5 a 7 afirmações em escala de Likert. Valide rigorosamente as frases garantindo que não haja duas perguntas na mesma sentença, ambiguidades ou induções morais óbvias.

Forneça também um título e uma descrição para o questionário.
"""


def questionnaire_agent_call(state: AgentState) -> dict:
    identified_risks = state.get("identified_risks", [])
    risk_classification = state.get("risk_classification", "")
    executive_summary = state.get("executive_summary", "")

    risks_text = "\n".join(f"- {risk}" for risk in identified_risks)

    human_content = (
        f"Classificação geral de risco do projeto: {risk_classification}\n\n"
        f"Sumário executivo:\n{executive_summary}\n\n"
        f"Riscos específicos identificados no projeto de IA:\n{risks_text}"
    )

    structured_llm = model.with_structured_output(QuestionnaireResult)
    result = structured_llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human_content)]
    )

    if isinstance(result, dict):
        result_obj = QuestionnaireResult(**result)
    else:
        result_obj = result

    summary = f"# {result_obj.title}\n\n{result_obj.description}\n\n"  # type: ignore
    for i, item in enumerate(result_obj.items, 1):  # type: ignore
        summary += f"**{i}.** {item.statement}\n"
        for opt in item.options:
            summary += f"   - {opt}\n"
        summary += "\n"

    return {
        "messages": [SystemMessage(content=summary)],
        "questionnaire_items": [item.model_dump() for item in result_obj.items],  # type: ignore
        "llm_calls": 1,
    }
