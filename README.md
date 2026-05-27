# ✈️ AI Travel Concierge Agent

A premium, production-ready conversational AI Travel Agent built with **Streamlit**, **LangGraph**, and **LangChain**. The application provides an intelligent, end-to-end itinerary builder, live flight schedules, hotel lodging metrics, multi-day weather telemetry, and real-time restaurant review mining—engineered to be highly accessible and deeply intuitive for all tiers of users.

---

## 🌟 Key Features

* **📌 Unified AI Agent Core:** Driven by Google's `gemini-2.5-flash` model utilizing a robust, version-proof LangGraph ReAct execution pool.
* **✈️ Real-Time Flight Schedules:** Integrated with SerpAPI to pull live departure/arrival wall-clock hours, carrier flight numbers, and actual fares in INR.
* **🏨 Dynamic Hotel Search:** Tracks open properties, exact nightly room breakdown rates, star rankings, and key perks natively on local maps.
* **🍽️ Restaurant Review Matrix:** Dedicated dining lookups that mine genuine customer feedback snippets, specialties, and location coordinates instantly.
* **🌤️ Multi-Day Weather Telemetry:** Real-time temperature checks paired with structured 3-day look-ahead forecasts leveraging deep API streams.
* **🧠 Adaptive NLP Architecture:** Handles highly complex, multi-layered travel budget constraints while remaining simple enough to auto-generate placeholders for brief queries (e.g., "Hotels in Goa").
* **🌙 Premium Interface Design:** Elegant layout featuring dynamic card layouts, clean markdown data tables, and a polished, single-click dark mode switch.

---

## 🛠️ Tech Stack & Architecture

* **Frontend UI:** Streamlit (Python)
* **Agent Framework:** LangGraph (`create_react_agent`)
* **LLM Integration:** LangChain Google GenAI (`ChatGoogleGenerativeAI`)
* **Execution Core:** Self-contained, robust single-file lifecycle architecture (`app.py`) to bypass microservice caching issues.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure you have Python 3.10+ installed on your local development machine.

### 2. Installation
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/your-username/travel_agent.git](https://github.com/your-username/travel_agent.git)
cd travel_agent
pip install streamlit requests langchain-core langchain-google-genai langgraph pydantic
