import os
import torch
import re
import numpy as np
import nltk
import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from rank_bm25 import BM25Okapi

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')
nltk.data.path.append(os.path.expanduser("~/.cache/nltk_data"))

# Set up GPU availability check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

### 1. Data Collection & Preprocessing

def load_and_preprocess_data_from_github(repo_url, file_extensions=[".txt"]):
    """Loads and preprocesses financial text data from a GitHub repository."""
    documents = []
    filenames = []
    
    # Example GitHub repo raw URL: "https://raw.githubusercontent.com/username/repo/main/"
    for ext in file_extensions:
        api_url = f"{repo_url}?ref=main"
        response = requests.get(api_url)
        
        if response.status_code == 200:
            files = response.json()  # Assuming GitHub API returns JSON data with file URLs
            for file in files:
                if file['name'].endswith(ext):
                    file_url = file['download_url']
                    text_response = requests.get(file_url)
                    if text_response.status_code == 200:
                        text = re.sub(r'\s+', ' ', text_response.text).strip()
                        documents.append(text)
                        filenames.append(file['name'])
                    else:
                        print(f"Failed to download file: {file['name']}")
        else:
            print(f"Failed to access GitHub repo: {repo_url}")
    
    print(f"Loaded {len(documents)} documents from GitHub.")
    return documents, filenames

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
    
    print(f"Chunked text into {len(overlapped_chunks)} chunks.")
    return overlapped_chunks

### 2. Basic RAG Implementation

def embed_text(texts, model_name="all-MiniLM-L6-v2"):
    """Embeds text using Sentence Transformers."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    return np.array(embeddings).reshape(-1, embeddings.shape[-1])

def retrieve_relevant_chunks_embedding(query_embedding, document_embeddings, chunks, top_k=3):
    """Retrieves relevant chunks based on embedding similarity."""
    # Reshape query_embedding to be 2D if it's a single vector
    query_embedding = np.atleast_2d(query_embedding)

    # Check the shapes of the embeddings for debugging
    print(f"Query embedding shape: {query_embedding.shape}")
    print(f"Document embeddings shape: {document_embeddings.shape}")

    similarities = cosine_similarity(query_embedding, document_embeddings)[0]
    relevant_indices = np.argsort(similarities)[::-1][:top_k]
    return [chunks[i] for i in relevant_indices], similarities[relevant_indices]

### 3. Advanced RAG Implementation (Multi-Stage Retrieval)

def retrieve_relevant_chunks_bm25(query, documents, top_k=3):
    """Retrieves relevant chunks using BM25."""
    if len(documents) == 0:
        return [], []
    
    tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
    
    if len(tokenized_docs) == 0:
        return [], []
    
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = word_tokenize(query.lower())
    doc_scores = bm25.get_scores(tokenized_query)
    relevant_indices = np.argsort(doc_scores)[::-1][:top_k]
    
    return [documents[i] for i in relevant_indices], doc_scores[relevant_indices]

def rerank_chunks(query_embedding, chunk_embeddings, chunks, top_k=3):
    """Reranks chunks based on embedding similarity."""
    # Reshape query_embedding to ensure it's 2D
    query_embedding = np.atleast_2d(query_embedding)

    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    relevant_indices = np.argsort(similarities)[::-1][:top_k]
    print(f"Reranked {top_k} chunks.")  # debug
    return [chunks[i] for i in relevant_indices], similarities[relevant_indices]

def generate_response(query, relevant_chunks):
    """Generates a response using retrieved chunks."""
    return " ".join(relevant_chunks)

### 4. Guard Rail Implementation

def is_valid_financial_query(query):
    """Validates if the query is a financial question."""
    financial_keywords = ["revenue", "profit", "earnings", "expenses", "balance sheet", "cash flow", "financial", "growth", "debt", "assets", "liabilities"]
    return any(keyword in query.lower() for keyword in financial_keywords)

### 5. Streamlit Implementation

def set_dark_mode_and_styling():
    """Injects custom CSS for dark mode and styling."""
    custom_css = """
    <style>
        body { background-color: #1E1E1E; color: white; }
        .stTextInput input, .stTextArea textarea {
            background-color: #333; color: white; border-radius: 10px;
        }
        .stButton button { background-color: #4CAF50; color: white; }
        .answer-field, .confidence-field { width: 50% !important; margin: 0 auto; }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def handle_query(query, all_chunks, chunk_embeddings, documents):
    """Handles user query."""
    if not is_valid_financial_query(query):
        st.error("Invalid query. Please ask a financial question.")
        return None, None, None, None
    
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

def streamapplicationmain(all_chunks, chunk_embeddings, documents):
    set_dark_mode_and_styling()
    if 'answer' not in st.session_state:
        st.session_state.answer = "Awaiting query..."
    if 'confidence' not in st.session_state:
        st.session_state.confidence = 0.0

    st.title("Financial Query Chatbot")
    st.subheader("Ask your financial queries, and get answers instantly!")

    query = st.text_input("Enter your financial query:")

    if st.button("Get Answer"):
        response, embedding_scores, bm25_scores, reranked_scores = handle_query(query, all_chunks, chunk_embeddings, documents)
        
        if response:
            st.session_state.answer = response
            st.session_state.confidence = reranked_scores[0] if reranked_scores.size > 0 else 0.0

    st.markdown('<div class="answer-field">', unsafe_allow_html=True)
    st.text_area("Answer", value=st.session_state.answer, height=150)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="confidence-field">', unsafe_allow_html=True)
    st.text_area("Confidence Level", value=f"{st.session_state.confidence * 100:.2f}%", height=100)
    st.markdown('</div>', unsafe_allow_html=True)

### 6. Main Function

def main():
    github_repo_url = "https://api.github.com/repos/username/repo/contents/"  # Replace with your GitHub repo URL
    documents, filenames = load_and_preprocess_data_from_github(github_repo_url)
    
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc))

    chunk_embeddings = embed_text(all_chunks)

    streamapplicationmain(all_chunks, chunk_embeddings, documents)

if __name__ == "__main__":
    main()
