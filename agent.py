import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from tools import my_tools

def clean_agent_output(state):
    messages = state["messages"]
    if not messages:
        return state
        
    last_msg = messages[-1]
    if hasattr(last_msg, "content") and isinstance(last_msg.content, str):
        text = last_msg.content.strip()
        if '"text":' in text and '"extras"' in text:
            try:
                clean_json = text.lstrip("0123456789:[] ,\t\n").rstrip("]")
                if not clean_json.startswith("{") and "{" in clean_json:
                    clean_json = clean_json[clean_json.index("{"):]
                
                parsed = json.loads(clean_json)
                if "text" in parsed:
                    last_msg.content = parsed["text"]
            except Exception:
                if '"text":"' in text:
                    extracted = text.split('"text":"', 1)[1].split('","extras"', 1)[0]
                    last_msg.content = extracted.rstrip('"\n\t }')
                elif '"text": "' in text:
                    extracted = text.split('"text": "', 1)[1].split('",\n"extras"', 1)[0]
                    last_msg.content = extracted.rstrip('"\n\t }')
                    
    return {"messages": messages}

def get_agent():
    # Force stable and text-optimized core engine
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    # --- CRITICAL SYSTEM INSTRUCTIONS OVERHAUL ---
    # This structure strictly dictates how the agent handles weather queries every single time.
    system_instructions = (
        "You are an elite, professional AeroConcierge AI travel assistant. "
        "Whenever a user asks about the weather, temperature, or conditions of a city/location, "
        "DO NOT just give a simple one-line reading or temperature number. "
        "You MUST structure your final response exactly in the following professional manner:\n\n"
        
        "1. VISUAL FORECAST LINE:\n"
        "Provide a quick, readable row matching this exact syntax pattern for each day:\n"
        "Day Name | Rain % | Min Temp / Max Temp (e.g., 'Mon | 5% | 32°C / 43°C')\n\n"
        
        "2. DETAILED 6-DAY FORECAST SUMMARY:\n"
        "Provide a comprehensive, itemized breakdown detailing conditions for Today and the next 5 days. "
        "Include Expected Highs/Lows, UV Index, and Sky conditions.\n\n"
        
        "3. CRITICAL ENVIRONMENTAL PRECAUTIONS:\n"
        "Provide a distinct section labeled '🚨 Critical Heatwave Precautions' or '🚨 Safety Precautions' "
        "containing detailed advice on hydration, avoiding peak sun exposure, clothing, or weather risks "
        "tailored directly to the active climate conditions.\n\n"
        
        "Always keep your tone premium, corporate, and helpful. Use your tools to fetch data if needed."
    )

    agent = create_react_agent(llm, tools=my_tools, prompt=system_instructions)
    
    original_invoke = agent.invoke
    def secured_invoke(*args, **kwargs):
        result = original_invoke(*args, **kwargs)
        return clean_agent_output(result)
        
    agent.invoke = secured_invoke
    return agent
