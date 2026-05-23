from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from tools import my_tools

def get_agent():
    # 1. Initialize Gemini with temperature=0 for focused, reliable tool calls
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    # 2. Define the core instructions as a system prompt string
    system_instructions = (
        "You are a helpful travel assistant. Use your tools to look up real-time weather "
        "or current web information if your context documents do not contain the answers."
    )

    # 3. Create the LangGraph agent
    # FIXED: Swapped 'state_modifier' for 'prompt' to match modern LangGraph versions
    agent = create_react_agent(llm, tools=my_tools, prompt=system_instructions)
    
    return agent
