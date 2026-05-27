def get_keys_pool():
    if "GEMINI_API_KEYS" not in st.secrets:
        return []
    
    raw_keys = st.secrets["GEMINI_API_KEYS"]
    
    # If the user configured it as a true TOML array list
    if isinstance(raw_keys, list):
        return [str(k).strip() for k in raw_keys if str(k).strip()]
        
    # If it is a comma-separated single string layer
    try:
        # Clean out any accidental stray single/double quotes or brackets added by mistake
        cleaned_string = str(raw_keys).replace("[", "").replace("]", "").replace('"', '').replace("'", "")
        return [k.strip() for k in cleaned_string.split(",") if k.strip()]
    except Exception:
        return []
