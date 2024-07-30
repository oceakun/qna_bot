import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL")

def get_response(query):
    try:
        response = requests.post(API_URL, json={"query": query})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")
        return None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(layout="wide")
st.title("Medical QA Chatbot")

st.markdown("""
<style>
    .chat-container {
        display: flex;
        flex-direction: column;
        height: calc(100vh - 200px);
        overflow-y: auto;
        padding: 10px;
    }
    .input-box {
        position: fixed;
        bottom: 0;
        width: 100%;
        background: white;
        padding: 10px;
        border-top: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])

user_query = st.chat_input("Type your question here:")

if user_query:
    st.session_state.chat_history.append({'role': 'user', 'content': user_query})
    
    with st.chat_message("user"):
        st.write(user_query)
    
    with st.spinner("Getting response..."):
        result = get_response(user_query)
        if result:
            answers = result.get('data', [])
            if answers:
                answer_text = answers[0]['answer']
                st.session_state.chat_history.append({'role': 'assistant', 'content': answer_text})
                
                with st.chat_message("assistant"):
                    st.write(answer_text)
            else:
                st.error("No answer found.")
        else:
            st.error("Failed to get a response.")