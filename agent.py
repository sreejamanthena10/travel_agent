import json
import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from tools import my_tools

def clean_agent_output(result):
    if not result or "messages" not in result:
        return result
    messages = result["messages"]
    if not messages:
        return result
    last_msg = messages[-1]
    if hasattr(last_msg, "content"):
        if isinstance(last_msg.content, list):
            extracted = []
            for chunk in last_msg.content:
                if isinstance(chunk, dict) and "text" in chunk:
                    extracted.append(chunk["text"])
                elif isinstance(chunk, str):
                    extracted.append(chunk)
            last_msg.content = "\n".join(extracted)
            
        if isinstance(last_msg.content, str):
            text = last_msg.content.strip()
            if "'text':" in text or '"text":' in text:
                for anchor in ["'text': '", '"text": "', "'text':", '"text":']:
                    if anchor in text:
                        try:
                            text = text.split(anchor, 1)[1]
                            for term in ["', 'extras'", '", "extras"', "',\n'extras'", '",\n"extras"']:
                                if term in text:
                                    text = text.split(term, 1)[0]
                            break
                        except:
                            pass
            last_msg.content = text.replace(r'\"', '"').rstrip("'\" \n\t}]")
    return result

def get_agent():
    """
    Safely parses the comma-separated key string and initializes the agent.
    Loops through backup options if a 429 quota or 400 error strikes.
    """
    keys_pool = []
    
    # Clean string extraction to completely bypass TOML bracket parsing bugs
    if "GEMINI_API_KEYS" in st.secrets:
        raw_keys = st.secrets["GEMINI_API_KEYS"]
        # Split by comma and strip out any accidental whitespace or quotes
        keys_pool = [k.strip().replace('"', '').replace("'", "") for k in raw_keys.split(",") if k.strip()]
    elif "GEMINI_API_KEY" in st.secrets:
        keys_pool = [st.secrets["GEMINI_API_KEY"].strip()]

    if not keys_pool:
        return None

    # Step-through key validation loop
    for current_key in keys_pool:
        try:
            # Skip obviously broken or placeholder entries
            if not current_key.startswith("AIzaSy"):
                continue
                
            os.environ["GOOGLE_API_KEY"] = current_key
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

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
        except Exception:
            continue # Try next key string if connection initialization errors out
            
    return None
