import os
import torch
import re
import numpy as np
import nltk
import streamlit as st
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

def load_and_preprocess_data():
    """Predefined financial text data."""
    content = """
    Accelerated growth momentum across businesses led to strong quarterly results
    Delivered highest ever Q1 consolidated profit, with strong performance across businesses
    Consolidated1 Revenue grew 26% year on year to Rs.5,859 Crore
    Consolidated PAT grew 42% year on year to Rs.429 Crore
    Active customer base at 39 million (up 55% year on year) aided by focus on granular retail growth across all businesses
    On track to deliver ahead of long term guidance (FY24) across businesses

    (Rs. crore)

    Consolidated results
     	Quarter 1
    Particulars	FY 22	FY 23	 
    Revenue1	4,632	5,859	26%
    Profit after tax (after minority interest)	302	429	42%

    Mumbai: Aditya Birla Capital Limited (“The Company”) announced its unaudited financial results for the quarter ended 30th June 2022.
    """
    documents = [content]
    filenames = ["financial_report.txt"]
    print(f"Loaded {len(documents)} documents.")
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
            overlapped_chunks.append(chunks[i] + " " + chunks[i+1][:overlap])
        else:
            overlapped_chunks.append(chunks[i])
    
    return overlapped_chunks

def preprocess_documents(documents):
    """Preprocesses documents."""
    preprocessed = []
    for doc in documents:
        doc = re.sub(r'\s+', ' ', doc)
        doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc)
        preprocessed.append(doc)
    return preprocessed

### 2. BM25 & Sentence Transformer Setup

def setup_search_model():
    """Sets up BM25 and Sentence Transformer."""
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)
    return model

def run_query(query, documents, model):
    """Runs a query against the documents."""
    # Preprocessing
    documents = preprocess_documents(documents)
    # Create BM25 object for term frequency-based search
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Query processing
    tokenized_query = word_tokenize(query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)

    # Rank by BM25 scores
    ranked_docs_by_bm25 = np.argsort(bm25_scores)[::-1]
    
    return ranked_docs_by_bm25, bm25_scores

def get_top_documents(query, documents, model, top_k=3):
    """Retrieve top documents based on query and BM25."""
    ranked_docs, bm25_scores = run_query(query, documents, model)
    top_docs = []
    for idx in ranked_docs[:top_k]:
        top_docs.append(documents[idx])
    
    return top_docs

### 3. Streamlit UI for User Interaction

def query_documents(query, documents, model):
    """Display query results using Streamlit."""
    # Retrieve top documents
    top_docs = get_top_documents(query, documents, model, top_k=3)
    
    st.write("### Query Results")
    for idx, doc in enumerate(top_docs):
        st.write(f"**Document {idx+1}:**")
        st.write(doc)

def main():
    """Run the Streamlit application."""
    # Load and preprocess data
    documents, filenames = load_and_preprocess_data()
    model = setup_search_model()

    # Streamlit Input
    st.title("Financial Query Chatbot")
    query = st.text_input("Enter your query:")

    if query:
        query_documents(query, documents, model)

if __name__ == "__main__":
    main()
