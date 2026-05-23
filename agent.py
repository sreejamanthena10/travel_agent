from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from tools import my_tools

def get_agent():
    # FIXED: Swapped model string to 'gemini-flash-latest' as requested
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

    # Core operational instructions for the travel assistant
    system_instructions = (
        "You are a helpful travel assistant. Use your tools to look up real-time weather "
        "or current web information if your context documents do not contain the answers."
    )

    # Build the modern LangGraph react agent instance
    agent = create_react_agent(llm, tools=my_tools, prompt=system_instructions)
    
    return agent
