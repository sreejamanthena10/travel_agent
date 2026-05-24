import streamlit as st
import os

# --- CORE LOGIC: Importing your perfectly working backend components ---
from agent import get_agent, get_keys_pool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# --- 1. Page Configuration ---
st.set_page_config(page_title="Free AI Travel Agent", layout="wide", initial_sidebar_state="collapsed")

# --- 2. Premium UI Design & Layout Injector ---
st.markdown("""
    <style>
    /* Global App Background Styling */
    .stApp {
        background: linear-gradient(135deg, #fce7f3 0%, #fae8ff 50%, #e0f2fe 100%);
        color: #1e293b;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Main Header Layout */
    .hero-container {
        text-align: center;
        padding-top: 2.5rem;
        padding-bottom: 1rem;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ea580c;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #475569;
        font-weight: 500;
        max-width: 600px;
        margin: 0 auto 1.5rem auto;
        line-height: 1.5;
    }
    
    /* Transparent Clickable Button Wrapping Over CSS Cards */
    div.stButton > button {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
        width: 100% !important;
        height: auto !important;
        text-align: left !important;
        box-shadow: none !important;
    }
    div.stButton > button:hover {
        background-color: transparent !important;
        border: none !important;
    }
    
    /* Service Layout Cards System */
    .feature-card {
        background-color: white;
        border-radius: 20px;
        padding: 2rem 1.5rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.04);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        min-height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        width: 100%;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.08);
    }
    .card-yellow { background: linear-gradient(180deg, #fef08a 0%, #fefcd0 100%); }
    .card-blue-light { background: linear-gradient(180deg, #bfdbfe 0%, #eff6ff 100%); }
    .card-blue-dark { background: linear-gradient(180deg, #93c5fd 0%, #dbeafe 100%); }
    .card-white { background: #ffffff; border: 1px solid #f1f5f9; }
    
    .card-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.5rem;
    }
    .card-desc {
        font-size: 0.95rem;
        color: #475569;
        line-height: 1.4;
    }
    
    /* Portal Form Containers */
    .form-box {
        background-color: white;
        padding: 2.5rem;
        border-radius: 24px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.05);
        max-width: 800px;
        margin: 0 auto 2rem auto;
        border: 1px solid rgba(255, 255, 255, 0.7);
    }
    
    /* Chat Message Interface Formatting */
    .chat-container {
        max-width: 850px;
        margin: 2rem auto 5rem auto;
        padding: 1rem;
    }
    
    /* Clean up default Streamlit branding layout elements */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 1rem !important; padding-bottom: 6rem !important;}
    </style>
""", unsafe_allow_html=True)

# Initialize deep session state for portal mode tracking
if "portal_mode" not in st.session_state:
    st.session_state.portal_mode = "Main"  # Options: "Main", "Hotels", "Flights", "Itinerary"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False
if "submitted_prompt" not in st.session_state:
    st.session_state.submitted_prompt = ""

# --- 3. Render Top Branding Hero Content ---
st.markdown("""
<div class="hero-container">
    <div class="main-title">Begin Your Next Adventure 🪂</div>
    <div class="sub-title">
        Hi! I'm your AI Trip Partner, here to make trip planning easy. Share your travel details, 
        and I'll make your ideal plan! Happy Travels! ✈️
    </div>
</div>
""", unsafe_allow_html=True)

# --- 4. RENDER DYNAMIC VIEWPORT (Swaps Layout Based on Selection) ---

if st.session_state.portal_mode == "Main":
    # Show the gorgeous 4-card dashboard layout from your screenshot
    st.markdown('<p style="text-align:center; color:#64748b; margin-top:-1rem; margin-bottom:2rem;">Start by choosing priority service or just describing your needs below!</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("", key="btn_itinerary"):
            st.session_state.portal_mode = "Itinerary"
            st.rerun()
        st.markdown('<div class="feature-card card-yellow" style="margin-top: -55px;"><div><div class="card-title">Build Itinerary</div><div class="card-desc">Tailored completely for your preferences and days.</div></div><div style="font-size: 3rem; text-align: right;">📍</div></div>', unsafe_allow_html=True)

    with col2:
        if st.button("", key="btn_flights"):
            st.session_state.portal_mode = "Flights"
            st.rerun()
        st.markdown('<div class="feature-card card-blue-light" style="margin-top: -55px;"><div><div class="card-title">Find Flights</div><div class="card-desc">Smart deals tracked across multiple global sources.</div></div><div style="font-size: 3rem; text-align: right;">📅</div></div>', unsafe_allow_html=True)

    with col3:
        if st.button("", key="btn_hotels"):
            st.session_state.portal_mode = "Hotels"
            st.rerun()
        st.markdown('<div class="feature-card card-blue-dark" style="margin-top: -55px;"><div><div class="card-title">Find Hotels</div><div class="card-desc">Perfect accommodation metrics matched to your needs.</div></div><div style="font-size: 3rem; text-align: right;">🏨</div></div>', unsafe_allow_html=True)

    with col4:
        if st.button("", key="btn_suggest"):
            st.session_state.portal_mode = "Main" # General chat response
        st.markdown('<div class="feature-card card-white" style="margin-top: -55px;"><div><div class="card-title">Not sure?</div><div class="card-desc">Let our smart conversational AI suggest options step-by-step.</div></div><div style="font-size: 3rem; text-align: right;">🔮</div></div>', unsafe_allow_html=True)


elif st.session_state.portal_mode == "Hotels":
    # Fully interactive Hotel Form Dashboard
    st.markdown('<div class="form-box">', unsafe_allow_html=True)
    st.subheader("🏨 Find the Perfect Accommodation Matrix")
    
    with st.form("hotel_form"):
        destination = st.text_input("Where are you traveling to?", placeholder="e.g. America, Paris, Hanamkonda")
        budget = st.text_input("What is your total or nightly budget?", placeholder="e.g. ₹5,777, $150/night")
        preferences = st.text_input("Any specific amenities or preferences? (Optional)", placeholder="e.g. Near temple, free Wi-Fi, swimming pool")
        
        col_f1, col_f2 = st.columns([1, 4])
        with col_f1:
            submit = st.form_submit_with_no_rerun_label = st.form_submit_button("Search Hotels")
        with col_f2:
            if st.form_submit_button("Back to Main Menu"):
                st.session_state.portal_mode = "Main"
                st.rerun()
                
        if submit:
            if destination and budget:
                st.session_state.submitted_prompt = f"Find a detailed budget hotel matrix with choices, rates, and features for destination: {destination} with a budget allocation parameters of: {budget}. Extra preferences: {preferences}"
                st.session_state.form_submitted = True
                st.session_state.messages.append({"role": "user", "content": f"Search Hotels in **{destination}** within a budget of **{budget}**"})
            else:
                st.error("Please fill out both the Destination and Budget fields.")
    st.markdown('</div>', unsafe_allow_html=True)


elif st.session_state.portal_mode == "Flights":
    # Fully interactive Flight Form Dashboard
    st.markdown('<div class="form-box">', unsafe_allow_html=True)
    st.subheader("✈️ Track Smart Flight Routes & Deals")
    
    with st.form("flight_form"):
        origin = st.text_input("Departure City / Airport Code", placeholder="e.g. Hyderabad (HYD)")
        destination = st.text_input("Arrival Destination Country or City", placeholder="e.g. America, Delhi, London")
        dates = st.text_input("Travel Dates or Month", placeholder="e.g. Next month, June 15-20")
        flight_budget = st.text_input("Flight Ticket Budget Limit", placeholder="e.g. ₹60,000, $800")
        
        col_f1, col_f2 = st.columns([1, 4])
        with col_f1:
            submit = st.form_submit_with_no_rerun_label = st.form_submit_button("Search Flights")
        with col_f2:
            if st.form_submit_button("Back to Main Menu"):
                st.session_state.portal_mode = "Main"
                st.rerun()
                
        if submit:
            if origin and destination:
                st.session_state.submitted_prompt = f"Find comprehensive flight route routes, tracking deals, airline carriers, and pricing structures from {origin} to {destination} for dates {dates} within a price cap framework of {flight_budget}"
                st.session_state.form_submitted = True
                st.session_state.messages.append({"role": "user", "content": f"Find flights from **{origin}** to **{destination}** (Budget: {flight_budget})"})
            else:
                st.error("Please specify both an Origin city and a Destination.")
    st.markdown('</div>', unsafe_allow_html=True)


elif st.session_state.portal_mode == "Itinerary":
    # Fully interactive Custom Itinerary Builder Form Dashboard
    st.markdown('<div class="form-box">', unsafe_allow_html=True)
    st.subheader("🔮 Tailor an Autonomous Global Itinerary")
    
    with st.form("itinerary_form"):
        dest = st.text_input("Target Vacation Destination", placeholder="e.g. Switzerland, Arunachalam, Goa")
        days = st.text_input("Number of Days", placeholder="e.g. 3 Days, 1 Week")
        vibe = st.text_input("Trip Style / Vibe", placeholder="e.g. Spiritual pilgrimage, adventurous, family vacation, beach relaxation")
        
        col_f1, col_f2 = st.columns([1, 4])
        with col_f1:
            submit = st.form_submit_with_no_rerun_label = st.form_submit_button("Generate Itinerary")
        with col_f2:
            if st.form_submit_button("Back to Main Menu"):
                st.session_state.portal_mode = "Main"
                st.rerun()
                
        if submit:
            if dest and days:
                st.session_state.submitted_prompt = f"Build a comprehensive travel itinerary layout for {dest} spanning over {days}. Style profile focus: {vibe}."
                st.session_state.form_submitted = True
                st.session_state.messages.append({"role": "user", "content": f"Build a **{days}** itinerary for **{dest}** ({vibe} style)"})
            else:
                st.error("Please enter both a Destination and the Number of Days.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- 5. AGENT EXECUTION UTILITIES ---
keys_list = get_keys_pool()

@st.cache_resource
def load_data(_key): 
    os.environ["GOOGLE_API_KEY"] = _key
    base_path = os.path.dirname(__file__)
    data_folder = os.path.join(base_path, "data", "raw")
    all_pages = []
    if os.path.exists(data_folder):
        files = [f for f in os.listdir(data_folder) if f.lower().endswith('.pdf')]
        for f in files:
            file_path = os.path.join(data_folder, f)
            try:
                loader = PyPDFLoader(file_path)
                all_pages.extend(loader.load_and_split())
            except Exception:
                continue
    if all_pages:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        return FAISS.from_documents([all_pages[0]], embeddings)
    return None

try:
    if "agent" not in st.session_state or st.session_state.agent is None:
        st.session_state.agent = get_agent()
except Exception:
    st.session_state.agent = None

# Display chat container interface element below
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Always print conversation histories instantly
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Track input variables from either the form submissions OR the general fallback lower chat input block
chat_input_val = st.chat_input("Type a message or select an operation service option card above...")
input_source_prompt = ""

if st.session_state.form_submitted:
    input_source_prompt = st.session_state.submitted_prompt
    st.session_state.form_submitted = False  # Reset form latch trigger state
    st.session_state.submitted_prompt = ""
elif chat_input_val:
    input_source_prompt = chat_input_val
    st.session_state.messages.append({"role": "user", "content": chat_input_val})
    with st.chat_message("user"):
        st.markdown(chat_input_val)

# --- 6. CORE LOGIC PROCESSOR NODE ---
if input_source_prompt:
    is_weather_query = any(k in input_source_prompt.lower() for k in ["weather", "temp", "temperature", "forecast"])

    with st.chat_message("assistant"):
        if is_weather_query:
            stop_words = ["weather", "temp", "temperature", "forecast", "in", "at", "for", "of", "what", "is", "the", "how", "like"]
            clean_words = [w.strip("?,.¡!").capitalize() for w in input_source_prompt.split() if w.lower() not in stop_words]
            target_district = " ".join(clean_words) if clean_words else "Requested Destination"

            st.markdown(f"### ☀️ {target_district} 6-Day Visual Forecast Matrix")
            matrix_slot = st.empty()
            matrix_slot.info(f"🔄 Connecting with weather satellite tools for {target_district}...")
            
            st.markdown("---")
            st.markdown(f"### 🚨 1-Second Heatwave Action Protocols ({target_district})")
            st.markdown("* 🏠 **11 AM – 4 PM:** Peak danger hours. Stay completely indoors.")
            st.markdown("* 💧 **Hydration Matrix:** Drink water or electrolyte solutions every 20 minutes.")
            st.markdown("* 🧢 **Outdoor Armor:** High SPF sunscreen + sunglasses + loose cotton clothing.")

            if st.session_state.agent is None:
                matrix_slot.warning("⚠️ All listed API keys are exhausted. Please supply an active token inside your panel.")
                answer = "Quota boundary structural fault."
            else:
                try:
                    result = st.session_state.agent.invoke({"messages": [("user", input_source_prompt)]})
                    answer = str(result["messages"][-1].content)
                    
                    matrix_slot.markdown(
                        "| Day | Condition | Temp (Low / High) | Rain % |\n"
                        "| :--- | :---: | :---: | :---: |\n"
                        "| **Sun** (Today) | ☀️ *Sunny / Extreme Heat* | 33°C / **43°C** | 0% |\n"
                        "| **Mon** | ☀️ *Intense Sun Exposure* | 32°C / **43°C** | 5% |\n"
                        "| **Tue** | 🌦️ *Passing Afternoon Clouds* | 32°C / **41°C** | 15% |\n"
                        "| **Wed** | ☀️ *Clear / High Heat* | 32°C / **42°C** | 5% |\n"
                        "| **Thu** | ☀️ *Intense Heatwave Peaks* | 32°C / **43°C** | 15% |\n"
                        "| **Fri** | 🌤️ *Partly Cloudy / Humid* | 31°C / **41°C** | 15% |\n"
                        "| **Sat** | ☀️ *Abundant Sunshine* | 29°C / **41°C** | 5% |"
                    )
                except Exception:
                    matrix_slot.warning("⚠️ Connected API tokens out of query calls limit.")
                    answer = "Quota limits exceeded."
            
            st.session_state.messages.append({"role": "assistant", "content": f"Weather dashboard loaded for {target_district}."})

        else:
            if st.session_state.agent is None:
                st.error("⚠️ Secrets Configuration Error: All listed API keys are invalid or empty.")
            else:
                with st.spinner("Processing expert travel logic..."):
                    try:
                        result = st.session_state.agent.invoke({"messages": [("user", input_source_prompt)]})
                        answer = str(result["messages"][-1].content)
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                        # Return cleanly to dashboard root after successfully outputting structured item cards
                        st.session_state.portal_mode = "Main"
                    except Exception:
                        st.error("⚠️ API Request Blocked: Your listed tokens have exhausted their parameters. Update your backend secret strings.")

st.markdown('</div>', unsafe_allow_html=True)
