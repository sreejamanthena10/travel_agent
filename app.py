# --- FIXED: Robust LangGraph Message Parser Node ---
                    result = live_agent.invoke({"messages": [("user", refined_query)]})
                    
                    # Extract the true human-readable AI message from the graph state array
                    agent_messages = result.get("messages", [])
                    answer = ""
                    
                    # Scan backwards from the end of the conversation history to find the text response
                    for msg in reversed(agent_messages):
                        # Verify the node belongs to the assistant and holds string data
                        if msg.type == "ai" and hasattr(msg, "content") and str(msg.content).strip():
                            answer = str(msg.content)
                            break
                    
                    # Fallback guard if formatting filters cleared out text content
                    if not answer and agent_messages:
                        answer = str(agent_messages[-1].content)

                    # Stream text to front end layout seamlessly
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
