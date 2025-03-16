import streamlit as st
import requests
import base64

# GitHub repository details
username = "VigneshKumarMuruganantham"
repository = "Streamlitapplicationforbits"
file_path = ""  # Leave it empty if you want to fetch files from the root folder
branch = "main"  # Ensure the branch is set to 'main'
access_token = "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"  # GitHub Personal Access Token

# GitHub API URL for accessing the folder contents in the 'main' branch
github_api_url = f"https://api.github.com/repos/{username}/{repository}/contents/{file_path}?ref={branch}"

# Set up the headers for authentication
headers = {
    "Authorization": f"token {access_token}"
}

# Function to fetch file list from the repository (in the 'main' branch)
def fetch_file_list():
    response = requests.get(github_api_url, headers=headers)
    if response.status_code == 200:
        return response.json()  # Returns list of files from the root folder in the 'main' branch
    else:
        st.error(f"Failed to fetch file list. Status code: {response.status_code}")
        return []

# Function to fetch file content from GitHub
def fetch_file_content(file_url):
    response = requests.get(file_url, headers=headers)
    if response.status_code == 200:
        file_data = response.json()
        # Decode the file content from base64
        file_content = base64.b64decode(file_data['content']).decode('utf-8')
        return file_content
    else:
        st.error("Failed to fetch file content.")
        return ""

# Streamlit UI
def display_files():
    file_list = fetch_file_list()
    if not file_list:
        return

    # Display the file names and sample content
    for file in file_list:
        file_name = file['name']
        if file_name.endswith('.txt'):
            st.write(f"### File: {file_name}")

            # Fetch and display the sample content of the file
            file_content = fetch_file_content(file['download_url'])
            sample_data = file_content[:500]  # Displaying first 500 characters as a sample

            st.write(f"**Sample Data:**")
            st.write(sample_data)

# Run the Streamlit app
if __name__ == "__main__":
    st.title("Financial Data Files")
    display_files()
