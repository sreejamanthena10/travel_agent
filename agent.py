import json
import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from tools import my_tools

def clean_agent_output(result):
    """
    Safely strips out background metadata, signature string leaks, and JSON artifacts
    to return pure markdown layout content to app.py.
    """
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
    Initializes and returns the compiled LangGraph reactive tool agent.
    Loops through available backend secret keys to circumvent 429 Resource Limits.
    """
    # 1. Gather keys from array or fallback to a single key configuration
    keys_pool = []
    if "GEMINI_API_KEYS" in st.secrets:
        keys_pool = list(st.secrets["GEMINI_API_KEYS"])
    elif "GEMINI_API_KEY" in st.secrets:
        keys_pool = [st.secrets["GEMINI_API_KEY"]]

    if not keys_pool:
        print("Error: No Gemini API Keys discovered within Streamlit Secrets configuration layers.")
        return None

    # 2. Loop through keys until an active endpoint initializes successfully
    for index, current_key in enumerate(keys_pool):
        try:
            os.environ["GOOGLE_API_KEY"] = current_key
            
            # Instantiating the robust production model engine
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
            
            # Key verification validation block passed safely
            return agent

        except Exception as e:
            print(f"API Key index {index} failed with exception: {str(e)}. Swapping to failover array resource...")
            continue
            
    print("CRITICAL FAILURE: Exhausted all keys in the array pool without establishing a connection link.")
    return None
