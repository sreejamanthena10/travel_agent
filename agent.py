import json
import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from tools import my_tools

def clean_agent_output(result):
    """
    Safely extracts pure markdown text and strips metadata/signature leaks
    to ensure the UI displays clean content immediately.
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

def get_keys_pool():
    """Extracts a verified list of keys from the raw comma-separated secrets string."""
    keys_pool = []
    if "GEMINI_API_KEYS" in st.secrets:
        raw_keys = st.secrets["GEMINI_API_KEYS"]
        keys_pool = [k.strip().replace('"', '').replace("'", "") for k in raw_keys.split(",") if k.strip()]
    elif "GEMINI_API_KEY" in st.secrets:
        keys_pool = [st.secrets["GEMINI_API_KEY"].strip()]
    return [k for k in keys_pool if k.startswith("AIzaSy")]

def get_agent():
    """Initializes the compiled agent by cycling through the keys pool validation layer."""
    keys_pool = get_keys_pool()
    if not keys_pool:
        return None

    for current_key in keys_pool:
        try:
            os.environ["GOOGLE_API_KEY"] = current_key
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

            # UNIVERSAL INTELLIGENCE MATRIX
            system_instructions = (
                "You are the ultimate AeroConcierge AI Global Travel Expert. You specialize in high-speed scannability.\n"
                "You create 6-day weather grids, locate budget-matched hotels, and map out sights all over the world—"
                "from major international hubs to local regional districts.\n\n"
                
                "FAIL-SAFE INSTRUCTION RULE:\n"
                "If your online search tools fail, time out, return an empty response, or encounter restriction barriers, "
                "you MUST NOT return an error message, apology, or refusal statement. Instead, immediately use your extensive "
                "internal global knowledge base to generate highly accurate, comprehensive, and realistic estimated budget matrices "
                "and structured markdown layouts for the requested destination.\n\n"
                
                "CRITICAL DESIGN RULES:\n"
                "1. NEVER write long, dense walls of text. Users must understand your response in 1 second using clean symbols, markdown tables, and bullet grids.\n"
                "2. LOCAL RULES: If asked about regional spots near Hanamkonda, Karimnagar, or Warangal, immediately highlight famous landmarks "
                "(like the Thousand Pillar Temple, Warangal Fort, Bhadrakali Temple) using checkboxes and clean icons.\n"
                "3. BUDGET ALLOCATION: When asked for accommodations anywhere in the world, split hotels into exact price tiers:\n"
                "   - 🎒 Budget Tiers (Hostels, local stays, pocket-friendly homestays)\n"
                "   - 🏨 Mid-tier Tiers (Standard comfort hotels, family stays)\n"
                "   - 💎 Luxury Tiers (Premium 5-Star luxury properties, high-end resorts)\n"
                "4. WEATHER CHECKS: If asked about weather or temperatures, output the results using a clean day-by-day table structure.\n\n"
                
                "EXPECTED FORMAT TEMPLATES:\n\n"
                "If requested SIGHTSEEING:\n"
                "### 🗺️ Top Landmarks Near [Destination Name]\n"
                "- 🏛️ **[Landmark Name]** | *Best Time: 4 PM - 7 PM* | Quick 1-sentence highlight of features.\n\n"
                
                "If requested HOTELS:\n"
                "### 🏨 Accommodation Matrix: [Destination Name]\n"
                "| Class | Recommended Stay | Est. Nightly Rate | Vibe & Key Feature |\n"
                "| :--- | :--- | :--- | :--- |\n"
                "| 🎒 Budget | Stay Name Here | ₹ / $ Amount | Budget-matched, clean, great reviews |"
            )

            agent = create_react_agent(llm, tools=my_tools, prompt=system_instructions)
            original_invoke = agent.invoke
            def secured_invoke(*args, **kwargs):
                return clean_agent_output(original_invoke(*args, **kwargs))
                
            agent.invoke = secured_invoke
            return agent
        except Exception:
            continue
            
    return None
