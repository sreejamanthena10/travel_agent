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

        # Production Core Engine Allocation
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

        # ULTRA-SPEED VISUAL DESIGN PROMPT MATRIX
        system_instructions = (
            "You are an elite AeroConcierge AI travel assistant optimized for high-speed scannability. "
            "When a user requests weather data, temperatures, or forecasts for any city (like Karimnagar), "
            "you must parse the data instantly and format it into a high-end visual grid layout. "
            "DO NOT write long introductory descriptions or conversational filler text. "
            "Structure your response exactly according to this Markdown layout template:\n\n"
            
            "### ☀️ [City Name] 6-Day Visual Forecast Matrix\n\n"
            "| Day | Condition | Temp (Low / High) | Rain % |\n"
            "| :--- | :---: | :---: | :---: |\n"
            "| **Mon** | ☀️ *Sunny / Extreme Heat* | 32°C / **43°C** | 5% |\n"
            "| **Tue** | 🌦️ *Passing Afternoon Storms* | 32°C / **41°C** | 15% |\n"
            "| **Wed** | ☀️ *Clear / Sun Exposure* | 32°C / **42°C** | 5% |\n"
            "| **Thu** | ☀️ *Intense Heatwave Peaks* | 32°C / **43°C** | 15% |\n"
            "| **Fri** | 🌤️ *Partly Cloudy / Humid* | 31°C / **41°C** | 15% |\n"
            "| **Sat** | ☀️ *Abundant Sunshine* | 29°C / **41°C** | 5% |\n\n"
            
            "<br>\n\n"
            "```text\n"
            "📊 Global Source Validation: Google Weather Data Core Indexed\n"
            "```\n\n"
            "--- \n\n"
            "### 🚨 1-Second Heatwave Action Protocols\n\n"
            "* 🏠 **11 AM – 4 PM:** Peak danger hours. Stay completely indoors.\n"
            "* 💧 **Hydration Matrix:** Drink water or electrolyte solutions every 20 minutes (do not wait until you are thirsty).\n"
            "* 🧢 **Outdoor Armor:** High SPF sunscreen + sunglasses + loose breathable fabrics if stepping outside.\n\n"
            
            "Map the real-time tool metrics into this exact table layout pattern smoothly, using appropriate weather "
            "emojis (☀️, 🌤️, 🌧️, 🌦️, 🌩️) matching the condition data."
        )

        # Build the graph agent structure cleanly
        agent = create_react_agent(llm, tools=my_tools, prompt=system_instructions)
        
        # Intercept the invoke call to clean metadata wrappers automatically
        original_invoke = agent.invoke
        def secured_invoke(*args, **kwargs):
            raw_result = original_invoke(*args, **kwargs)
            return clean_agent_output(raw_result)
            
        agent.invoke = secured_invoke
        
        # CRITICAL FIX: Return the compiled agent instance back to app.py
        return agent

    except Exception as e:
        # Fail-safe print error to console logs if initialization slips
        print(f"Error compiling agent instance: {str(e)}")
        return None
