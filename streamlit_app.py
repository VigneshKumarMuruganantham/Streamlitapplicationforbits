import streamlit as st
import requests
import json
import base64
import pandas as pd

st.set_page_config(page_title="Financial Data Viewer", layout="wide")

st.title("Financial Data Viewer")
st.markdown("This application displays financial data from text files stored in GitHub.")

# GitHub credentials and repository information
username = "VigneshKumarMuruganantham"
repository = "Streamlitapplicationforbits"
file_path = "financial_data"
branch = "main"
access_token = st.secrets["github_token"]  # Store this in Streamlit secrets when deploying

# Function to get file names from the GitHub repository
def get_file_names():
    github_api_url = f"https://api.github.com/repos/{username}/{repository}/contents/{file_path}?ref={branch}"
    
    headers = {
        "Authorization": f"token {access_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        response = requests.get(github_api_url, headers=headers)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        files = response.json()
        txt_files = [file['name'] for file in files if file['name'].endswith('.txt')]
        
        return txt_files
    except requests.exceptions.RequestException as e:
        st.error(f"Error accessing GitHub API: {str(e)}")
        return []

# Function to read file content from GitHub
def get_file_content(file_name):
    file_url = f"https://api.github.com/repos/{username}/{repository}/contents/{file_path}/{file_name}?ref={branch}"
    
    headers = {
        "Authorization": f"token {access_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        response = requests.get(file_url, headers=headers)
        response.raise_for_status()
        
        content_data = response.json()
        if 'content' in content_data:
            # GitHub API returns base64 encoded content
            file_content = base64.b64decode(content_data['content']).decode('utf-8')
            return file_content
        else:
            return "No content found in the file."
    except requests.exceptions.RequestException as e:
        return f"Error reading file: {str(e)}"

# Get available .txt files
txt_files = get_file_names()

if not txt_files:
    st.warning("No .txt files found in the specified GitHub folder.")
else:
    st.sidebar.header("Available Files")
    
    # Display all file names in the sidebar
    st.sidebar.subheader("All Text Files")
    for file_name in txt_files:
        st.sidebar.text(file_name)
    
    # File selection
    selected_file = st.sidebar.selectbox("Select a file to view", txt_files)
    
    # Display the selected file content
    if selected_file:
        st.header(f"Contents of {selected_file}")
        file_content = get_file_content(selected_file)
        
        # Display raw data
        with st.expander("Raw Data", expanded=True):
            st.text(file_content)
        
        # Try to parse as structured data (if possible)
        try:
            # Try to parse as CSV-like data
            lines = file_content.strip().split('\n')
            if len(lines) > 1:
                # Check if tab-separated
                if '\t' in lines[0]:
                    delimiter = '\t'
                # Check if comma-separated
                elif ',' in lines[0]:
                    delimiter = ','
                # Default to space
                else:
                    delimiter = None
                
                df = pd.read_csv(
                    pd.StringIO(file_content), 
                    delimiter=delimiter, 
                    error_bad_lines=False,
                    warn_bad_lines=False
                )
                
                st.subheader("Data Preview")
                st.dataframe(df)
                
                st.subheader("Summary Statistics")
                numeric_cols = df.select_dtypes(include=['number']).columns
                if not numeric_cols.empty:
                    st.dataframe(df[numeric_cols].describe())
                else:
                    st.info("No numeric columns found for statistics.")
        except Exception as e:
            st.info("The file doesn't appear to be in a structured format for data analysis.")
            st.text(f"Parsing error: {str(e)}")
Last edited 1 minute ago
