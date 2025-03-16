import os
import torch
import re
import numpy as np
import nltk
import streamlit as st
import requests
import base64
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from rank_bm25 import BM25Okapi

# Ensure NLTK resources are available
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.data.path.append(os.path.expanduser("~/.cache/nltk_data"))

# Set up GPU availability check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.sidebar.info(f'Using device: {device}')

### GitHub Integration
def load_data_from_github():
    """Load financial text data from GitHub repository."""
    username = "VigneshKumarMuruganantham"
    repository = "Streamlitapplicationforbits"
    file_path = "financial_data"
    branch = "main"
    
    # Store access token in Streamlit secrets for security
    # For development, you can set it directly, but NEVER commit this to version control
    # In production, use: access_token = st.secrets["github_token"]
    access_token = st.secrets.get("github_token", "")  # Use empty string as fallback for development

    # GitHub API URL for accessing the folder contents
    github_api_url = f"https://api.github.com/repos/{username}/{repository}/contents/{file_path}?ref={branch}"
    
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Add authorization header if token is available
    if access_token:
        headers["Authorization"] = f"token {access_token}"
    
    documents = []
    filenames = []
    
    try:
        # Get list of files in the directory
        response = requests.get(github_api_url, headers=headers)
        response.raise_for_status()
        files = response.json()
        
        # Filter for .txt files
        txt_files = [file for file in files if file['name'].endswith('.txt')]
        
        for file in txt_files:
            # Get file content
            file_url = file['download_url']
            file_content_response = requests.get(file_url, headers=headers)
            file_content_response.raise_for_status()
            
            # Preprocess text
            text = re.sub(r'\s+', ' ', file_content_response.text).strip()
            documents.append(text)
            filenames.append(file['name'])
        
        st.sidebar.success(f"Loaded {len(documents)} documents from GitHub.")
        return documents, filenames
    
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Error accessing GitHub: {str(e)}")
        return [], []

### Data Preprocessing
def chunk_text(text, chunk_size=512, overlap=50):
    """Chunks text into smaller pieces with overlap."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    overlapped_chunks = []
    for i in range(len(chunks)):
        if i + 1 < len(chunks):
            overlapped_chunks.append(chunks[i] + " " + " ".join(chunks[i+1].split(" ")[:overlap]))
        else:
            overlapped_chunks.append(chunks[i])
    
    return overlapped_chunks

### RAG Implementation
def embed_text(texts, model_name="all-MiniLM-L6-v2"):
    """Embeds text using Sentence Transformers."""
    model = SentenceTransformer(model_name)
    with st.spinner("Generating embeddings..."):
        return model.encode(texts)

def retrieve_relevant_chunks_embedding(query_embedding, document_embeddings, chunks, top_k=3):
    """Retrieves relevant chunks based on embedding similarity."""
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    relevant_indices = np.argsort(similarities)[::-1][:top_k]
    return [chunks[i] for i in relevant_indices], similarities[relevant_indices]

def retrieve_relevant_chunks_bm25(query, documents, top_k=3):
    """Retrieves relevant chunks using BM25."""
    tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = word_tokenize(query.lower())
    doc_scores = bm25.get_scores(tokenized_query)
    relevant_indices = np.argsort(doc_scores)[::-1][:top_k]
    return [documents[i] for i in relevant_indices], doc_scores[relevant_indices]

def rerank_chunks(query_embedding, chunk_embeddings, chunks, top_k=3):
    """Reranks chunks based on embedding similarity."""
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    relevant_indices = np.argsort(similarities)[::-1][:top_k]
    return [chunks[i] for i in relevant_indices], similarities[relevant_indices]

def generate_response(query, relevant_chunks):
    """Generates a response using retrieved chunks."""
    return " ".join(relevant_chunks)

### Guard Rail Implementation
def is_valid_financial_query(query):
    """Validates if the query is a financial question."""
    financial_keywords = ["revenue", "profit", "earnings", "expenses", "balance sheet", 
                         "cash flow", "financial", "growth", "debt", "assets", "liabilities"]
    return any(keyword in query.lower() for keyword in financial_keywords)

### UI Implementation
def set_dark_mode_and_styling():
    """Injects custom CSS for dark mode and styling."""
    custom_css = """
    <style>
    body {
        background-color: #1E1E1E;
        color: white;
    }
    .stTextInput input, .stTextArea textarea {
        background-color: #333;
        color: white;
        border-radius: 10px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
    }
    .answer-field, .confidence-field {
        width: 50% !important;
        margin: 0 auto;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def handle_query(query, all_chunks, chunk_embeddings, documents):
    """Handles user query."""
    if not is_valid_financial_query(query):
        st.error("Invalid query. Please ask a financial question.")
        return None, None, None, None
    
    with st.spinner("Processing query..."):
        # BM25 retrieval
        bm25_chunks, bm25_scores = retrieve_relevant_chunks_bm25(query, documents)
        
        # Embedding-based retrieval
        query_embedding = embed_text([query])[0]
        embedding_chunks, embedding_scores = retrieve_relevant_chunks_embedding(query_embedding, chunk_embeddings, all_chunks)
        
        # Reranking
        reranked_chunks, reranked_scores = rerank_chunks(query_embedding, embed_text(embedding_chunks), embedding_chunks)
        
        # Generate response
        response = generate_response(query, reranked_chunks)
        
    return response, embedding_scores, bm25_scores, reranked_scores

def display_file_samples(documents, filenames):
    """Display file names and sample content."""
    st.sidebar.header("Document Samples")
    for filename, content in zip(filenames, documents):
        with st.sidebar.expander(f"File: {filename}"):
            st.write(content[:300] + "...")  # Show first 300 chars

def streamapplicationmain():
    """Main Streamlit application."""
    set_dark_mode_and_styling()
    
    st.title("Financial Query Chatbot")
    st.subheader("Ask your financial queries, and get answers instantly!")
    
    # Load documents from GitHub
    with st.spinner("Loading documents from GitHub..."):
        documents, filenames = load_data_from_github()
    
    if not documents:
        st.error("No documents were loaded. Please check GitHub repository access.")
        return
    
    # Process documents
    with st.spinner("Processing documents..."):
        all_chunks = []
        for doc in documents:
            all_chunks.extend(chunk_text(doc))
        
        # Cache embeddings to improve performance
        if 'chunk_embeddings' not in st.session_state:
            st.session_state.chunk_embeddings = embed_text(all_chunks)
        chunk_embeddings = st.session_state.chunk_embeddings
    
    # Initialize session state
    if 'answer' not in st.session_state:
        st.session_state.answer = "Awaiting query..."
    if 'confidence' not in st.session_state:
        st.session_state.confidence = 0.0
    
    # Display file samples in sidebar
    display_file_samples(documents, filenames)
    
    # Query interface
    query = st.text_input("Enter your financial query:")
    
    if st.button("Get Answer"):
        response, embedding_scores, bm25_scores, reranked_scores = handle_query(
            query, all_chunks, chunk_embeddings, documents
        )
        
        if response:
            st.session_state.answer = response
            st.session_state.confidence = reranked_scores[0] if reranked_scores.size > 0 else 0.0
    
    st.markdown('<div class="answer-field">', unsafe_allow_html=True)
    st.text_area("Answer", value=st.session_state.answer, height=150)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="confidence-field">', unsafe_allow_html=True)
    st.text_area("Confidence Level", value=f"{st.session_state.confidence * 100:.2f}%", height=100)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Debugging information
    with st.expander("Debug Information"):
        st.write(f"Number of documents: {len(documents)}")
        st.write(f"Number of chunks: {len(all_chunks)}")
        st.write(f"Embedding dimensions: {chunk_embeddings.shape if hasattr(chunk_embeddings, 'shape') else 'N/A'}")

### Main Function
def main():
    st.set_page_config(
        page_title="Financial RAG Chatbot",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    streamapplicationmain()

if __name__ == "__main__":
    main()
