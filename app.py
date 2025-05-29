import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

# --- Load .env for environment variables ---
load_dotenv('.env')

# --- Header section ---
st.image("HKA_LOGO.png", width=700)
st.title("Devi Dayal's Own AI Chatbot")

# --- Define base URL and available models ---
base_url = "http://localhost:11434"
available_models = [
    "llama3.2:3b",
    "llama3.2:1b",
    "Sheldon:latest",
    "deepseek-r1:1.5b"
]

# --- Sidebar for user ID and model selection ---
st.sidebar.title("Configuration")
user_id = st.sidebar.text_input("Enter your user id:", "Devi Dayal")
selected_model = st.sidebar.selectbox("Choose a model", available_models)

# --- Function to get chat history for session ---
def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///chat_history.db")

# --- Clear history on button click ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.sidebar.button("🗑️ Start New Conversation"):
    st.session_state.chat_history = []
    history = get_session_history(user_id)
    history.clear()

# --- Show previous chat messages ---
for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# --- Set up LLM with selected model ---
llm = ChatOllama(base_url=base_url, model=selected_model)

# --- Prompt template ---
system = SystemMessagePromptTemplate.from_template("You are a helpful assistant.")
human = HumanMessagePromptTemplate.from_template("{input}")
messages = [system, MessagesPlaceholder(variable_name='history'), human]
prompt = ChatPromptTemplate(messages=messages)
chain = prompt | llm | StrOutputParser()

# --- Wrap with history ---
runnable_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='history'
)

# --- Define chat function with generator ---
def chat_with_llm(session_id, user_input):
    for output in runnable_with_history.stream({'input': user_input}, config={'configurable': {'session_id': session_id}}):
        yield output

# --- Chat input and response ---
user_input = st.chat_input("Ask something...")
if user_input:
    st.session_state.chat_history.append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="HKA_LOGO_ASS.PNG"):
        response = st.write_stream(chat_with_llm(user_id, user_input))

    st.session_state.chat_history.append({'role': 'assistant', 'content': response})
