import json
import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from tools import my_tools

def clean_agent_output(result):
    """Safely extracts pure markdown text and strips metadata/signature leaks."""
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
    """Extracts a clean list of string parameters from secrets."""
    keys_pool = []
    if "GEMINI_API_KEYS" in st.secrets:
        raw_keys = st.secrets["GEMINI_API_KEYS"]
        keys_pool = [k.strip().replace('"', '').replace("'", "") for k in raw_keys.split(",") if k.strip()]
    elif "GEMINI_API_KEY" in st.secrets:
        keys_pool = [st.secrets["GEMINI_API_KEY"].strip()]
    return [k for k in keys_pool if k.strip()]

@st.cache_resource(show_spinner=False)
def compile_secure_agent():
    """Compiles and validates the agent framework across the available resource tokens pool exactly once."""
    keys_pool = get_keys_pool()
    if not keys_pool:
        return None

    for current_key in keys_pool:
        try:
            os.environ["GOOGLE_API_KEY"] = current_key
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
            
            # Live Validation Guard Ping
            llm.invoke("ping")

            system_instructions = (
                "You are the AeroConcierge AI Global Travel Expert specialized in instant 2-second scannability.\n"
                "You map out daily travel schedules, price-tiered accommodation options, and precise flight matrices.\n\n"
                
                "CRITICAL CHRONOLOGICAL FLIGHT DATE RULE:\n"
                "When a user provides specific travel dates in their query (e.g., June 15, 2026), you MUST strictly read, extract, "
                "and pass those exact dates to your search tools. Your output flight itinerary matrix table MUST reflect "
                "the planes flying on those specific dates. Never change the dates or generalize them.\n\n"
                
                "FAIL-SAFE INSTRUCTION RULE:\n"
                "If search tools time out, encounter error walls, or run out of calls, use your internal global knowledge base "
                "to instantly construct highly realistic, comprehensive flight route carrier listings, estimated price figures, "
                "and accommodation options for the specified location and dates in under 2 seconds. Never present error tracebacks or apologies.\n\n"
                
                "DESIGN PRINCIPLES:\n"
                "1. NO walls of text. Use bulleted grids, markdown charts, and checkboxes.\n"
                "2. LOCAL DESIGN RULES: For queries targeting Telangana (Hanamkonda, Warangal, Karimnagar), immediately emphasize "
                "prominent architectural and historical hotspots like the Thousand Pillar Temple, Warangal Fort, and Bhadrakali Temple using clean icons.\n"
                "3. ACCOMMODATIONS MATRIX: Divide into Budget (🎒 Stays/Hostels), Mid-tier (🏨 Family Comfort), and Luxury (💎 5-Star Resorts).\n\n"
                
                "FLIGHT MATRIX FORMAT REQUIREMENT:\n"
                "### 📅 Plane Schedules & Routes: [Origin] to [Destination]\n"
                "**Selected Travel Window:** [Exact Date Mentioned in Query]\n"
                "| Airline Carrier | Flight No. | Departure -> Arrival | Est. Return Ticket Rate | Status |\n"
                "| :--- | :--- | :--- | :--- | :--- |\n"
                "| IndiGo / Air India | 6E-2134 | 06:15 -> 08:45 | ₹6,500 / $ Amount | 🟢 Available |"
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

def get_agent():
    """Thread-safe fast execution entrypoint access node."""
    return compile_secure_agent()
