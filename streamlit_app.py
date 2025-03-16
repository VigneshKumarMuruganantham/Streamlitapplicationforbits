def load_and_preprocess_data_from_github(repo_url, file_extensions=[".txt"]):
    """Loads and preprocesses financial text data from a GitHub repository."""
    documents = []
    filenames = []
    
    # Example GitHub repo raw URL: "https://raw.githubusercontent.com/username/repo/main/"
    for ext in file_extensions:
        api_url = f"{repo_url}?ref=main"
        response = requests.get(api_url)
        
        if response.status_code == 200:
            # Parse files
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

def streamapplicationmain(all_chunks, chunk_embeddings, documents, filenames):
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
    
    # Debugging output to check if documents and filenames are being populated correctly
    st.subheader("Loaded Files and Document Snippets")
    if len(filenames) == 0 or len(documents) == 0:
        st.write("No files loaded. Check your source data.")
    else:
        for i, filename in enumerate(filenames):
            st.write(f"**File**: {filename}")
            st.write(f"**First 200 characters**: {documents[i][:200]}...")  # Print the first 200 characters as a preview
            st.markdown("-" * 50)

### Main Function

def main():
    github_repo_url = "https://api.github.com/repos/username/repo/contents/"  # Replace with your GitHub repo URL
    documents, filenames = load_and_preprocess_data_from_github(github_repo_url)
    
    # Add debugging print to check loaded documents and filenames
    print(f"Documents: {documents}")
    print(f"Filenames: {filenames}")
    
    if len(documents) == 0 or len(filenames) == 0:
        print("No documents found. Check your GitHub repo access.")
    
    # Chunking the documents
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc))

    chunk_embeddings = embed_text(all_chunks)

    streamapplicationmain(all_chunks, chunk_embeddings, documents, filenames)

if __name__ == "__main__":
    main()
