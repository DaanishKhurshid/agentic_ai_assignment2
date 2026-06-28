import streamlit as st
import os
import shutil
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import TypedDict, List, Dict
from dotenv import load_dotenv

load_dotenv()

# --- 1. State Definition ---
class GraphState(TypedDict):
    original_query: str
    documents: List[str]
    generation: str

# --- 2. Setup (Cached to prevent DB locks) ---
@st.cache_resource
def get_system():
    # Clear the old DB to prevent 'InternalError'
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")
        
    # --- YOUR CONFIRMED WORKING MODELS ---
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")
    # -------------------------------------
    
    with open("orion_hub_manual.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    docs = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).create_documents([text])
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="chroma_db")
    return vectorstore.as_retriever(), llm

retriever, llm = get_system()

# --- 3. Nodes ---
def retrieve_node(state: GraphState) -> Dict:
    docs = retriever.invoke(state["original_query"])
    return {"documents": [d.page_content for d in docs]}

def generate_node(state: GraphState) -> Dict:
    context = "\n\n".join(state["documents"])
    prompt = f"Answer based on context: {context}\n\nQuery: {state['original_query']}"
    return {"generation": llm.invoke(prompt).content}

# --- 4. Graph Construction ---
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
app = workflow.compile()

# --- 5. Streamlit App ---
st.title("Orion Tech Support Hub")
query = st.text_input("How can I help you?")

if query:
    with st.spinner("Processing..."):
        result = app.invoke({"original_query": query})
        st.write(result["generation"])