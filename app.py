import streamlit as st
from llm_agent import ConversationAgent

st.set_page_config(page_title="Jarvis")
st.title("Barbe-Quasar")

if "agent" not in st.session_state:
    st.session_state.agent = ConversationAgent()

agent = st.session_state.agent

for msg in agent.conversation_history:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

user_input = st.chat_input("Que veux tu moussaillon ?")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    response = agent.ask_llm(user_input)

    with st.chat_message("assistant"):
        st.markdown(response)