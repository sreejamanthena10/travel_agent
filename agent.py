import json
import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from tools import my_tools

def clean_agent_output(result):
    """
    Safely intercept and clean the message output dictionary from LangGraph.
    """
    if not result or "messages" not in result:
        return result
        
    messages = result["messages"]
    if not messages:
        return result
        
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
                    
    return result

def get_agent():
    """
    Initializes and returns the compiled LangGraph reactive tool agent.
    """
    try:
        # Verify API key availability prior to model binding
        if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in st.secrets:
            os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]

        # HIGH-QUOTA ENGINE ALLOCATION: Swapped to gemini-2.5-flash to bypass the 20 req/day limit
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

        # DYNAMIC DISTRICT DESIGN MATRIX PROMPT
        system_instructions = (
            "You are an elite AeroConcierge AI travel assistant optimized for high-speed layout scannability. "
            "When a user asks for the weather, temperature, or forecast of ANY district or city, "
            "you must use your tools to check the live metrics and output the results EXACTLY using "
            "the Markdown structure template below. Replace '[Insert District Name]' with the actual "
            "name of the district the user asked for. Do not add any conversational filler or introductions.\n\n"
            
            "### ☀️ [Insert District Name] 6-Day Visual Forecast Matrix\n\n"
            "| Day | Condition | Temp (Low / High) | Rain % |\n"
            "| :--- | :---: | :---: | :---: |\n"
            "| **Sun** (Today) | [Emoji] *[Live Condition]* | [Min]°C / **[Max]°C** | [Rain]% |\n"
            "| **Mon** | [Emoji] *[Live Condition]* | [Min]°C / **[Max]°C** | [Rain]% |\n"
            "| **Tue** | [Emoji] *[Live Condition]* | [Min]°C / **[Max]°C** | [Rain]% |\n"
            "| **Wed** | [Emoji] *[Live Condition]* | [Min]°C / **[Max]°C** | [Rain]% |\n"
            "| **Thu** | [Emoji] *[Live Condition]* | [Min]°C / **[Max]°C** | [Rain]% |\n"
            "| **Fri** | [Emoji] *[Live Condition]* | [Min]°C / **[Max]°C** | [Rain]% |\n"
            "| **Sat** | [Emoji] *[Live Condition]* | [Min]°C / **[Max]°C** | [Rain]% |\n\n"
            
            "<br>\n\n"
            "```text\n"
            "📊 Global Source Validation: Google Weather Data Core Indexed\n"
            "```\n\n"
            "--- \n\n"
            "### 🚨 1-Second Heatwave Action Protocols ([Insert District Name])\n\n"
            "* 🏠 **11 AM – 4 PM:** Peak danger hours. Stay completely indoors to avoid extreme ambient temperatures.\n"
            "* 💧 **Hydration Matrix:** Drink water, buttermilk, or electrolyte solutions every 20 minutes (do not wait until you feel thirsty).\n"
            "* 🧢 **Outdoor Armor:** High SPF sunscreen + sunglasses + a wide-brimmed hat + loose, light breathable cotton fabrics if stepping outside.\n\n"
            
            "Ensure that you choose appropriate weather emojis (☀️, 🌤️, 🌧️, 🌦️, 🌩️, 💨) that match the live fetched data "
            "metrics perfectly for each day."
        )

        # Build the graph agent structure cleanly
        agent = create_react_agent(llm, tools=my_tools, prompt=system_instructions)
        
        # Intercept the invoke call to clean metadata wrappers automatically
        original_invoke = agent.invoke
        def secured_invoke(*args, **kwargs):
            raw_result = original_invoke(*args, **kwargs)
            return clean_agent_output(raw_result)
            
        agent.invoke = secured_invoke
        
        return agent

    except Exception as e:
        print(f"Error compiling agent instance: {str(e)}")
        return None
