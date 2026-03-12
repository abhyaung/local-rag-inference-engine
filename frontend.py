import streamlit as st
import requests
import json

#Configuration
BACKEND_URL = "http://127.0.0.1:8000/generate"
st.set_page_config(page_title="Abhyaung's AI", page_icon="🤖")

#UI Header
st.title("Resume Analyzer - Local RAG AI System")
st.markdown("*Running on Linux Ubuntu | Llama-3 | FastAPI*")

#File Upload section
with st.sidebar:
    st.header("Uploade Resume")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Analyzing and remembering document..."):
                #Send the file to FastAPI
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                responses = requests.post("http://127.0.0.1:8000/upload", files=files)

                if responses.status_code == 200:
                    st.success("Document Processed! You can now ask questions")
                else:
                    st.error("Error: Failed to process the documents")

#SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []


#Display chat message
#Loop through previous messages and display them here
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#Input Box
#This waits for the user to type and hit enter
if prompt := st.chat_input("Ask me anything about Abhyaung..."):
    
    #1. Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role":"user","content":prompt})

    #2. Call the brain
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                #Send the POST Request to your FastAPI server
                payload = {"prompt":prompt}
                response = requests.post(BACKEND_URL, json=payload)

                if response.status_code == 200:
                    data = response.json()

                    #answer = f"System Status: {data}"
                    answer = data["response"]

                else:
                    answer = f"Error: {response.status_code}"

            except Exception as e:
                answer = f"Connection Failed: {e}"

            st.markdown(answer)
            st.session_state.messages.append({"role":"assistant", "content": answer})

