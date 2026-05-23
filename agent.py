from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from tools import my_tools

def get_agent():
    # 1. Initialize Gemini with temperature=0 for focused, reliable tool calls
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    # 2. Give the agent its clear system instructions
    instructions = (
        "You are a helpful travel assistant. Use your tools to look up real-time weather "
        "or current web information if your context documents do not contain the answers."
    )

    # 3. Create the modern, robust LangGraph agent
    # This automatically matches the {"messages": [("user", ...)]} structure used in your app.py!
    agent = create_react_agent(llm, tools=my_tools, state_modifier=instructions)
    
    return agent
