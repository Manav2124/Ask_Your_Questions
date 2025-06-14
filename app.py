
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize embeddings model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Set Streamlit page config
st.set_page_config(page_title="Document QA Assistant", layout="centered")

# --- Custom CSS for UI ---
custom_css = """
<style>
    body {
        background-color: #f4f6f8;
    }
    .main {
        font-family: 'Segoe UI', sans-serif;
    }
    .title-style {
        font-size: 2.5rem;
        font-weight: 700;
        color: #665bb8;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle-style {
        font-size: 1.2rem;
        color: #495057;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stFileUploader {
        background-color: #003b78 !important;
        border: 2px dashed #6c63ff;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .question-box {
        font-size: 1.1rem;
        color: #212529;
        line-height: 1.6;
        background-color: #ffffff;
        border: 1px solid #ced4da;
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    textarea {
        font-size: 1.05rem !important;
    }
    .stSpinner > div > div {
        color: #343a40;
        font-size: 1rem;
    }
    .stButton > button {
        background: linear-gradient(to right, #6c63ff, #5a4ec9);
        color: white;
        border: none;
        padding: 0.6em 1.2em;
        font-weight: 600;
        font-size: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Functions ---

def process_uploaded_file(uploaded_file):
    """Process uploaded PDF or text file and return chunks"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    try:
        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(file_path=tmp_path)
        else:
            loader = TextLoader(file_path=tmp_path)
        
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=400
        )
        return text_splitter.split_documents(documents=docs)
    finally:
        os.unlink(tmp_path)

def initialize_vector_store(docs):
    return QdrantVectorStore.from_documents(
        documents=docs,
        url="http://localhost:6333",
        collection_name="learning_vectors",
        embedding=embedding_model
    )

def get_answer(query, vector_db):
    search_results = vector_db.similarity_search(query=query)
    context = "\n\n".join([
        f"Page Content: {result.page_content}\nPage Number: {result.metadata.get('page_label', 'N/A')}"
        for result in search_results
    ])
    
    system_prompt = f"""
    You are a helpful AI Assistant who answers user queries based on the available context
    retrieved from a document. Provide detailed answers and include page references when available.

    Context:
    {context}
    """
    
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
    )
    return response.choices[0].message.content

# --- App Header ---
st.markdown('<div class="title-style">📄 Document QA Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-style">Upload a PDF or text file and ask questions about its content</div>', unsafe_allow_html=True)

# --- Session State ---
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'processed' not in st.session_state:
    st.session_state.processed = False

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])

if uploaded_file and not st.session_state.processed:
    with st.spinner("⏳ Processing document..."):
        chunks = process_uploaded_file(uploaded_file)
        st.session_state.vector_db = initialize_vector_store(chunks)
        st.session_state.processed = True
        st.success("✅ Document processed successfully! You can now ask questions.")

# --- Query Box & Answer ---
if st.session_state.processed:
    st.markdown("## ❓ Ask Your Question")
    user_query = st.text_area("Ask something about the document", height=100, placeholder="e.g., What does the introduction say?")
    send_button = st.button("📤 Send")

    if send_button and user_query.strip():
        with st.spinner("🤖 Thinking..."):
            answer = get_answer(user_query, st.session_state.vector_db)
            st.markdown(f"""
            <div class="question-box">
                <strong>Answer:</strong><br>{answer}
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("📂 Please upload a document to get started.")

# --- Reset Button ---
st.button("🔁 Reset Session", on_click=lambda: (st.session_state.clear(), st.rerun()))
