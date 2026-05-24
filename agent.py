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
        if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in st.secrets:
            os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]

        # High-quota, ultra-fast production model allocation
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

        # UNIFIED TRAVEL CONCIERGE DESIGN PROMPT MATRIX
        system_instructions = (
            "You are the ultimate AeroConcierge AI Global Travel Expert. You specialize in high-speed scannability. "
            "You create 6-day weather grids, locate budget-matched hotels, and map out sights all over the world—"
            "from major international cities to local regional districts.\n\n"
            
            "CRITICAL DESIGN RULES:\n"
            "1. NEVER write long, dense walls of text. Users must understand your response in 1 second using clean symbols and markdown grids.\n"
            "2. LOCAL RULES: If asked about regional spots (e.g., Hanamkonda, Karimnagar, Warangal), immediately highlight historic landmarks "
            "(like the Thousand Pillar Temple, Warangal Fort, Bhadrakali Temple) using checkboxes and clean icons.\n"
            "3. BUDGET ALLOCATION: When asked for accommodations anywhere in the world, split hotels into exact price tiers:\n"
            "   - 🎒 Budget Tiers (Hostels, local stays, pocket-friendly homestays)\n"
            "   - 🏨 Mid-tier Tiers (Standard comfort hotels, family stays)\n"
            "   - 💎 Luxury Tiers (Premium 5-Star luxury properties, high-end resorts)\n"
            "4. WEATHER CHECKS: If asked about weather, temperature, or forecasts, output the results using your strict 6-day grid format and safety protocols.\n\n"
            
            "EXPECTED FORMAT TEMPLATES:\n\n"
            "If requested LOCAL SIGHTSEEING, output instantly like this:\n"
            "### 🗺️ Top Landmarks Near [Destination Name]\n"
            "- 🏛️ **[Landmark Name]** | *Best Time: 4 PM - 7 PM* | Quick 1-sentence highlight of history or features.\n"
            "- 🌳 **[Landmark Name]** | *Best Time: Morning* | Quick 1-sentence highlight.\n\n"
            
            "If requested HOTELS/TRIPS under a budget, output instantly like this:\n"
            "### 🏨 Accommodation Matrix: [Destination Name]\n"
            "| Class | Recommended Stay | Est. Nightly Rate | Vibe & Key Feature |\n"
            "| :--- | :--- | :--- | :--- |\n"
            "| 🎒 Budget | Stay Name Here | ₹ / $ Amount | Budget-matched, clean, great reviews |\n"
            "| 🏨 Mid-tier | Stay Name Here | ₹ / $ Amount | Comfortable amenities, pool access |\n"
            "| 💎 Luxury | Stay Name Here | ₹ / $ Amount | Premium 5-star executive experience |\n\n"
            
            "If requested WEATHER, output instantly like this:\n"
            "### ☀️ [District Name] 6-Day Visual Forecast Matrix\n"
            "| Day | Condition | Temp (Low / High) | Rain % |\n"
            "| :--- | :---: | :---: | :---: |\n"
            "| **Sun** (Today) | ☀️ *Sunny / Extreme Heat* | 33°C / **43°C** | 0% |\n\n"
            "### 🚨 1-Second Heatwave Action Protocols\n"
            "* 🏠 **11 AM – 4 PM:** Peak danger hours. Stay indoors.\n\n"
            
            "Use your tools smoothly to verify real-time facts, local spots, and accurate global pricing data."
        )

        agent = create_react_agent(llm, tools=my_tools, prompt=system_instructions)
        
        original_invoke = agent.invoke
        def secured_invoke(*args, **kwargs):
            raw_result = original_invoke(*args, **kwargs)
            return clean_agent_output(raw_result)
            
        agent.invoke = secured_invoke
        
        return agent

    except Exception as e:
        print(f"Error compiling agent instance: {str(e)}")
        return None
