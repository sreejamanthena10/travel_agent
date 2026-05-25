import os
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
# Import your real live tools from your tools file
from tools import search_flights, search_hotels, get_weather, plan_itinerary

def get_keys_pool():
    """Safely extracts and parses the comma-separated key pool string from Streamlit Secrets."""
    if "GEMINI_API_KEYS" not in st.secrets:
        return []
    raw_keys = st.secrets["GEMINI_API_KEYS"]
    # Splits by comma and cleans away any accidental blank spaces or trailing lines
    return [k.strip() for k in raw_keys.split(",") if k.strip()]

def get_agent():
    """Cycles through the independent project keys pool to compile a warm, functional live agent thread."""
    keys_pool = get_keys_pool()
    
    if not keys_pool:
        print("❌ Critical Config Error: No keys found in GEMINI_API_KEYS secrets array.")
        return None

    # Loop through each separate project key in your secrets panel
    for current_key in keys_pool:
        try:
            # 1. Configure the core Google Generative AI bindings with the active key
            genai.configure(api_key=current_key)
            
            # 2. Initialize the official LangChain model instance explicitly wrapping that active key context
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=current_key,
                temperature=0.3
            )
            
            # 3. Assemble your real-time travel toolkit array mapping to your true tools file
            tools_list = [search_flights, search_hotels, get_weather, plan_itinerary]
            
            # 4. Compile the reactive state graph agent thread layout
            compiled_react_agent = create_react_agent(llm, tools=tools_list)
            
            # 5. Lightweight verification check: Test if this specific project key has free quota space left
            # We do a super fast execution check to see if Google accepts the call
            test_response = llm.invoke("Check connection state: respond with exactly one word.")
            
            if test_response and test_response.content:
                # If the verification check passes, this project key is completely healthy! Stop looking and return it.
                return compiled_react_agent
                
        except Exception as e:
            error_msg = str(e).lower()
            # If this key hits a 429 quota block or a bad token 400 error, skip it immediately and try the next project key string!
            if "429" in error_msg or "resource_exhausted" in error_msg or "invalid" in error_msg or "expired" in error_msg:
                print(f"⚠️ Key slot exhausted or invalid, cycling safely to next independent project key...")
                continue
            else:
                # If it's an unrelated code error, log it and keep testing the pool
                print(f"⚠️ Internal verification warning: {error_msg}")
                continue

    # If the loop completes and every single key in the secrets panel failed verification, return None
    return None
