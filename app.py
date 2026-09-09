"""
app.py
------
Optional Streamlit web UI for the AI Student Support Assistant.

Run:
    streamlit run app.py

Requires the same setup as main.py (Ollama running, `python ingest.py`
already run once).
"""

import uuid

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from agent import build_agent

st.set_page_config(page_title="AI Student Support Assistant", page_icon="🎓")

st.title("🎓 AI Student Support Assistant")
st.caption("RAG + Tools + Memory — answers grounded in regulations, syllabus, FAQs & notices")

with st.sidebar:
    st.header("Session")
    student_id = st.text_input("Student ID", value=st.session_state.get("student_id", ""))
    if st.button("Start / Reset Session"):
        st.session_state.student_id = student_id or "guest"
        st.session_state.thread_id = f"{st.session_state.student_id}-{uuid.uuid4().hex[:8]}"
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        "**Try asking:**\n"
        "- What is the minimum attendance required?\n"
        "- I attended 60 out of 90 classes, am I eligible for exams?\n"
        "- What topics are in the AI syllabus?\n"
        "- Any recent notices?\n"
        "- Remember that my program is B.Tech CSE\n"
    )

if "student_id" not in st.session_state:
    st.session_state.student_id = "guest"
    st.session_state.thread_id = f"guest-{uuid.uuid4().hex[:8]}"
    st.session_state.chat_history = []

if "agent" not in st.session_state:
    with st.spinner("Loading agent (LLM + vector store)..."):
        st.session_state.agent = build_agent()

for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)

user_input = st.chat_input("Ask about regulations, syllabus, FAQs, or notices...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    contextualized_input = f"[student_id={st.session_state.student_id}] {user_input}"
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.agent.invoke(
                    {"messages": [HumanMessage(content=contextualized_input)]},
                    config=config,
                )
                response = result["messages"][-1].content
            except Exception as e:
                response = (
                    f"⚠️ Error: {e}\n\nMake sure Ollama is running and models are pulled."
                )
        st.markdown(response)

    st.session_state.chat_history.append(("assistant", response))
