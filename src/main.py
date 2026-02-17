from .agents.risk_agent import risks_agent
from .agents.incident_agent import incident_agent

def main():
    print("Welcome to the AI Ethics Multi-Agent System!")
    print("This system allows you to analyze AI ethics incidents and risks using specialized agents.")
    print("You can ask the Incident Analysis Agent about specific incidents, or the Risk Analysis Agent about potential risks.")
    print("Let's get started!\n")

    while True:
        user_input = input("Enter your query (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        if "incident" in user_input.lower():
            print("\nAnalyzing incident...\n")
            response = incident_agent.invoke(user_input)
            print(f"Incident Analysis:\n{response}\n")
        elif "risk" in user_input.lower():
            print("\nAnalyzing risk...\n")
            response = risks_agent.invoke(user_input)
            print(f"Risk Analysis:\n{response}\n")
        else:
            print("Please specify whether you want to analyze an 'incident' or a 'risk'.\n")