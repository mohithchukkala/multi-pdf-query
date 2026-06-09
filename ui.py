import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Agentic RAG", layout="wide")

st.title("Agentic & Corrective RAG")

# Sidebar for Ingestion
with st.sidebar:
    st.header("Data Ingestion")
    uploaded_file = st.file_uploader("Upload SEC Filing (PDF)", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Ingest Document"):
            with st.spinner("Ingesting..."):
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                try:
                    response = requests.post(f"{API_URL}/ingest", files=files)
                    if response.status_code == 200:
                        st.success(response.json()["message"])
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Thinking..."):
            try:
                response = requests.post(f"{API_URL}/query", json={"question": prompt})
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    steps = data["steps"]
                    
                    # Show steps in expander
                    with st.expander("Agent Steps"):
                        for step in steps:
                            st.write(step)
                            
                    full_response = answer
                    message_placeholder.markdown(full_response)
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")
                
    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})
