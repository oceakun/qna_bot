import streamlit as st
import requests
import os
4e3
API_URL = os.getenv("API_URL")

def get_response(query):
    try:
        response = requests.post(API_URL, json={"query": query, "model":"llm"})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")
        return None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# st.set_page_config(layout="wide")
st.title("Medical QA Chatbot (llm)")

for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

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
                print("answers[0] : ", answers[0])
                answer_text = answers[0]['answer']
                confidence_score = answers[0]['confidence']

                content_to_display_0= "*Confidence score* : "+ str(round(answers[0]['confidence']*100, 3)) + "%<br>" +  answers[0]['answer']
                content_to_display_1= "*Confidence score* : "+ str(round(answers[1]['confidence']*100, 3)) + "%<br>" +  answers[1]['answer']
                content_to_display_2= "*Confidence score* : "+ str(round(answers[2]['confidence']*100, 3)) + "%<br>" +  answers[2]['answer']
                
                st.session_state.chat_history.append({'role': 'assistant', 'content':content_to_display_0})
                st.session_state.chat_history.append({'role': 'assistant', 'content':content_to_display_1})
                st.session_state.chat_history.append({'role': 'assistant', 'content':content_to_display_2})
                
                with st.chat_message("assistant"):
                    st.markdown(content_to_display_0, unsafe_allow_html=True)
                    st.markdown(content_to_display_1, unsafe_allow_html=True)
                    st.markdown(content_to_display_2, unsafe_allow_html=True)
            else:
                st.error("No answer found.")
        else:
            st.error("Failed to get a response.")