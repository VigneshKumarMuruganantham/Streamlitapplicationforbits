import os
import torch
import re
import numpy as np
import nltk
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

# Create a directory for NLTK data if it doesn't exist
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Set the NLTK data path explicitly
nltk.data.path.append(nltk_data_dir)

# Download required NLTK data with explicit download location
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)

# Set up GPU availability check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

### 1. Data Collection & Preprocessing

def load_and_preprocess_data(data_dir):
    """Loads and preprocesses financial text data from a directory."""
    documents = []
    filenames = []
    
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        st.warning(f"Created empty data directory: {data_dir}. Please add .txt files.")
        return [], []
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = re.sub(r'\s+', ' ', f.read()).strip()
                    documents.append(text)
                    filenames.append(filename)
            except Exception as e:
                st.error(f"Error reading {filename}: {e}")
    
    if not documents:
        st.warning(f"No .txt files found in {data_dir}.")
    else:
        print(f"Loaded {len(documents)} documents.")
    
    return documents, filenames

def chunk_text(text, chunk_size=512, overlap=50):
    """Chunks text into smaller pieces with overlap.
    Uses manual sentence splitting to avoid NLTK dependencies."""
    # Simple sentence splitting by punctuation
    sentences = []
    for sent in re.split(r'(?<=[.!?])\s+', text):
        if sent.strip():
            sentences.append(sent.strip())
    
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
            words = chunks[i+1].split()
            overlap_words = words[:min(overlap, len(words))]
            overlapped_chunks.append(chunks[i] + " " + " ".join(overlap_words))
        else:
            overlapped_chunks.append(chunks[i])
    
    print(f"Chunked text into {len(overlapped_chunks)} chunks.")
    return overlapped_chunks

### 2. Basic RAG Implementation

def embed_text(texts, model_name="all-MiniLM-L6-v2"):
    """Embeds text using Sentence Transformers."""
    if not texts:
        return np.array([])
    
    try:
        model = SentenceTransformer(model_name, device=device)
        return model.encode(texts)
    except Exception as e:
        st.error(f"Error embedding text: {e}")
        return np.array([])

def retrieve_relevant_chunks_embedding(query_embedding, document_embeddings, chunks, top_k=3):
    """Retrieves relevant chunks based on embedding similarity."""
    if len(document_embeddings) == 0 or len(chunks) == 0:
        return [], np.array([])
    
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    relevant_indices = np.argsort(similarities)[::-1][:min(top_k, len(similarities))]
    return [chunks[i] for i in relevant_indices], similarities[relevant_indices]

### 3. Advanced RAG Implementation (Multi-Stage Retrieval)

def simple_tokenize(text):
    """Simple tokenization function to avoid NLTK dependencies."""
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text.split()

def retrieve_relevant_chunks_bm25(query, documents, top_k=3):
    """Retrieves relevant chunks using BM25."""
    if not documents:
        return [], np.array([])
    
    try:
        # Use simple tokenization instead of NLTK's word_tokenize
        tokenized_docs = [simple_tokenize(doc) for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        tokenized_query = simple_tokenize(query)
        doc_scores = bm25.get_scores(tokenized_query)
        relevant_indices = np.argsort(doc_scores)[::-1][:min(top_k, len(doc_scores))]
        return [documents[i] for i in relevant_indices], doc_scores[relevant_indices]
    except Exception as e:
        st.error(f"Error in BM25 retrieval: {e}")
        import traceback
        st.text(traceback.format_exc())
        return [], np.array([])

def rerank_chunks(query_embedding, chunk_embeddings, chunks, top_k=3):
    """Reranks chunks based on embedding similarity."""
    if len(chunk_embeddings) == 0 or len(chunks) == 0:
        return [], np.array([])
    
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    relevant_indices = np.argsort(similarities)[::-1][:min(top_k, len(similarities))]
    print(f"Reranked {min(top_k, len(similarities))} chunks.")  # debug
    return [chunks[i] for i in relevant_indices], similarities[relevant_indices]

def generate_response(query, relevant_chunks, source_indices, source_filenames):
    """Generates a response using retrieved chunks with actual content."""
    if not relevant_chunks:
        return "No relevant information found. Please try a different query."
    
    # Generate a more comprehensive response including content from chunks
    response = "Based on my analysis of your financial query, here is the relevant information:\n\n"
    
    # Track which documents we've used
    used_documents = {}
    
    # Include content from each chunk with source attribution
    for i, (chunk, idx) in enumerate(zip(relevant_chunks, source_indices)):
        if idx < len(source_filenames):
            filename = source_filenames[idx]
            
            # Extract a shorter version of the chunk to avoid overwhelming output
            # Find sentences that have query terms for more focused responses
            query_terms = set(simple_tokenize(query))
            sentences = re.split(r'(?<=[.!?])\s+', chunk)
            
            relevant_sentences = []
            for sentence in sentences:
                sentence_tokens = set(simple_tokenize(sentence))
                # If sentence contains any query terms, include it
                if any(term in sentence_tokens for term in query_terms):
                    relevant_sentences.append(sentence)
            
            # If no relevant sentences found, use the first 1-2 sentences
            if not relevant_sentences and sentences:
                relevant_sentences = sentences[:min(2, len(sentences))]
            
            relevant_content = " ".join(relevant_sentences)
            
            # Truncate if still too long
            if len(relevant_content) > 300:
                relevant_content = relevant_content[:297] + "..."
            
            # Add to used documents
            if filename not in used_documents:
                used_documents[filename] = []
            
            if relevant_content not in used_documents[filename]:
                used_documents[filename].append(relevant_content)
    
    # Format the response with document titles and content
    for filename, contents in used_documents.items():
        response += f"From {filename}:\n"
        for content in contents:
            response += f"• {content}\n\n"
    
    response += "Please consult with a financial advisor for personalized advice based on this information."
    return response

def get_document_index(chunk, all_chunks, documents):
    """Get the document index for a given chunk."""
    # Find which document contains this chunk
    chunk_idx = -1
    try:
        chunk_idx = all_chunks.index(chunk)
    except ValueError:
        pass
        
    if chunk_idx == -1:
        # Direct matching failed, try substring matching
        for i, doc in enumerate(documents):
            if chunk in doc:
                return i
        return 0  # Default to first document if not found
        
    # Count chunks to determine source document
    count = 0
    for i, doc in enumerate(documents):
        doc_chunks = chunk_text(doc)
        count += len(doc_chunks)
        if chunk_idx < count:
            return i
    return 0  # Default to first document if not found

### 4. Guard Rail Implementation

def is_valid_financial_query(query):
    """Validates if the query is a financial question."""
    if not query.strip():
        return False
        
    financial_keywords = [
        "revenue", "profit", "earnings", "expenses", "balance", "cash flow", 
        "financial", "growth", "debt", "assets", "liabilities", "investment",
        "stock", "share", "market", "capital", "dividend", "equity", "fund",
        "portfolio", "interest", "return", "budget", "forecast", "fiscal",
        "tax", "income", "loss", "statement", "quarter", "annual", "report"
    ]
    
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
        .answer-field { width: 100% !important; margin: 0 auto; }
        .confidence-field { width: 50% !important; margin: 0 auto; }
        .stTextArea textarea { font-size: 16px; line-height: 1.5; }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def handle_query(query, all_chunks, chunk_embeddings, documents, filenames):
    """Handles user query."""
    if not query.strip():
        st.warning("Please enter a query.")
        return None, None, None, None
        
    if not is_valid_financial_query(query):
        st.warning("Your query doesn't appear to be financial in nature. Please ask a financial question.")
        return None, None, None, None
    
    if not documents or not all_chunks:
        st.error("No documents available. Please add .txt files to the financial_data directory.")
        return None, None, None, None
    
    with st.spinner("Processing your query..."):
        # BM25 retrieval
        bm25_chunks, bm25_scores = retrieve_relevant_chunks_bm25(query, documents)
        
        # Embedding-based retrieval
        query_embedding = embed_text([query])[0]
        
        # Get more chunks for better coverage
        embedding_chunks, embedding_scores = retrieve_relevant_chunks_embedding(query_embedding, chunk_embeddings, all_chunks, top_k=5)
        
        # Reranking
        reranked_chunks, reranked_scores = rerank_chunks(query_embedding, embed_text(embedding_chunks), embedding_chunks, top_k=3)
        
        # Get document indices for chunks
        source_indices = [get_document_index(chunk, all_chunks, documents) for chunk in reranked_chunks]
        
        # Generate response with actual content
        response = generate_response(query, reranked_chunks, source_indices, filenames)
    
    return response, embedding_scores, bm25_scores, reranked_scores

def streamapplicationmain(all_chunks, chunk_embeddings, documents, filenames):
    set_dark_mode_and_styling()
    
    if 'answer' not in st.session_state:
        st.session_state.answer = "Awaiting query..."
    if 'confidence' not in st.session_state:
        st.session_state.confidence = 0.0

    st.title("Financial Query Chatbot")
    st.subheader("Ask your financial queries, and get answers instantly!")

    # Show document status
    st.sidebar.header("Document Status")
    if not documents:
        st.sidebar.error("No documents loaded. Add .txt files to the financial_data directory.")
    else:
        st.sidebar.success(f"Loaded {len(documents)} documents with {len(all_chunks)} chunks.")
        # Display only document titles, not content
        st.sidebar.subheader("Available Documents")
        for filename in filenames:
            st.sidebar.text(f"- {filename}")

    query = st.text_input("Enter your financial query:")

    if st.button("Get Answer"):
        response, embedding_scores, bm25_scores, reranked_scores = handle_query(query, all_chunks, chunk_embeddings, documents, filenames)
        
        if response:
            st.session_state.answer = response
            st.session_state.confidence = reranked_scores[0] if len(reranked_scores) > 0 else 0.0

    st.markdown('<div class="answer-field">', unsafe_allow_html=True)
    st.text_area("Answer", value=st.session_state.answer, height=150)
    st.markdown('</div>', unsafe_allow_html=True)

    # Changed height from 50 to 70 (minimum required is 68)
    st.markdown('<div class="confidence-field">', unsafe_allow_html=True)
    confidence_value = f"{st.session_state.confidence * 100:.2f}%" if st.session_state.confidence else "N/A"
    st.text_area("Confidence Level", value=confidence_value, height=70)
    st.markdown('</div>', unsafe_allow_html=True)

    # Add debug section in the sidebar
    st.sidebar.header("Debug Information")
    if st.sidebar.checkbox("Show Debug Info"):
        st.sidebar.text(f"Using device: {device}")
        st.sidebar.text(f"NLTK data path: {nltk_data_dir}")

### 6. Main Function
def main():
    try:
        st.set_page_config(page_title="Financial Query Assistant", layout="wide")
        
        data_dir = "financial_data"  # Replace with your data directory
        documents, filenames = load_and_preprocess_data(data_dir)
        
        all_chunks = []
        for doc in documents:
            all_chunks.extend(chunk_text(doc))
        
        chunk_embeddings = embed_text(all_chunks)
        
        streamapplicationmain(all_chunks, chunk_embeddings, documents, filenames)
        
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.text("Please check the logs for more details.")
        import traceback
        st.text(traceback.format_exc())

if __name__ == "__main__":
    main()
