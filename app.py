else:
            live_agent = get_agent()
            if live_agent is None:
                st.error("❌ Secrets Configuration Error: All listed tokens are invalid, empty, or exhausted.")
            else:
                with st.spinner("Processing expert travel logic..."):
                    try:
                        date_match = re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(st|nd|rd|th)?(,\s+\d{4})?', user_input, re.IGNORECASE)
                        extracted_date_context = f" on date {date_match.group(0)}" if date_match else ""
                        refined_query = f"{user_input}{extracted_date_context}. Ensure all flight tables explicitly reflect active schedules matching this timestamp context parameters."
                        
                        # Execute framework graph state payload
                        result = live_agent.invoke({"messages": [("user", refined_query)]})
                        
                        # --- UNIVERSAL MESSAGE EXTRACTOR LAYER ---
                        agent_messages = result.get("messages", [])
                        answer = ""
                        
                        for msg in reversed(agent_messages):
                            # Check for class names, string types, or dictionary structures safely
                            msg_type = getattr(msg, "type", "").lower()
                            class_name = msg.__class__.__name__
                            
                            if "ai" in msg_type or "ai" in class_name.lower():
                                if hasattr(msg, "content") and str(msg.content).strip():
                                    answer = str(msg.content)
                                    break
                        
                        # Ultimate fallback: if the loop skipped it, take the last available message content
                        if not answer and agent_messages:
                            last_msg = agent_messages[-1]
                            if hasattr(last_msg, "content"):
                                answer = str(last_msg.content)
                            elif isinstance(last_msg, dict) and "content" in last_msg:
                                answer = str(last_msg["content"])
                            else:
                                answer = str(last_msg)

                        # Render response instantly onto the screen
                        if answer.strip():
                            st.markdown(answer)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                        else:
                            st.warning("⚠️ The agent processed your query but returned an empty text layer.")
                            
                    except Exception as e:
                        st.error(f"⚠️ Operational parameters limit reached: {str(e)}")
