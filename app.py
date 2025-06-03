import streamlit as st
import requests
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

# --- Load environment variables ---
load_dotenv('.env')

# --- Header section ---
st.image("HKA_LOGO.png", width=700)
st.title("Devi Dayal's Own AI Chatbot")

# --- Ollama Model Discovery ---
base_url = "http://localhost:11434"

def get_local_ollama_models():
    try:
        response = requests.get(f"{base_url}/api/tags")
        response.raise_for_status()
        tags = response.json().get("models", [])
        return [tag["name"] for tag in tags]
    except Exception as e:
        st.error(f"Could not fetch local models from Ollama: {e}")
        return []

AVAILABLE_MODELS = get_local_ollama_models()

# --- Sidebar Configuration ---
st.sidebar.title("Configuration")
user_id = st.sidebar.text_input("Enter your user id:", "Devi Dayal")

# Set default model to "llama3.2:1b" if available
default_model = "llama3.2:1b"
default_index = AVAILABLE_MODELS.index(default_model) if default_model in AVAILABLE_MODELS else 0

selected_model = st.sidebar.selectbox("Choose a model", AVAILABLE_MODELS, index=default_index)

word_limit = st.sidebar.slider("Select number of words", min_value=20, max_value=1000, value=100, step=10)

# Stop sequence inputs (2 fields)
st.sidebar.markdown("### Stop Sequences")
stop1 = st.sidebar.text_input("Stop Sequence 1", "")
stop2 = st.sidebar.text_input("Stop Sequence 2", "")
stop_sequences = [s for s in [stop1, stop2] if s.strip()]

# --- Session Chat History Setup ---
def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///chat_history.db")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.sidebar.button("🗑️ Start New Conversation"):
    st.session_state.chat_history = []
    history = get_session_history(user_id)
    history.clear()

# --- Display previous chat messages ---
for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# --- Initialize LLM with stop sequences ---
llm = ChatOllama(
    base_url=base_url,
    model=selected_model,
    stop=stop_sequences
)

# --- Prompt Template with History and Word Limit ---
system = SystemMessagePromptTemplate.from_template(
    "You are a helpful assistant. Please keep your answers concise and within approximately {word_limit} words."
)
human = HumanMessagePromptTemplate.from_template("{input}")
messages = [system, MessagesPlaceholder(variable_name='history'), human]
prompt = ChatPromptTemplate.from_messages(messages)
chain = prompt | llm | StrOutputParser()

# --- Wrap with history-aware runner ---
runnable_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='history'
)

# --- Chat Input and Streamed Response ---
user_input = st.chat_input("Ask something...")
if user_input:
    st.session_state.chat_history.append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="HKA_LOGO_ASS.PNG"):
        response = st.write_stream(
            runnable_with_history.stream(
                {'input': user_input, 'word_limit': word_limit},
                config={'configurable': {'session_id': user_id}}
            )
        )

    st.session_state.chat_history.append({'role': 'assistant', 'content': response})
