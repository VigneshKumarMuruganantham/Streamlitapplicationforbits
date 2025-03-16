import os
import streamlit as st

# Path to the folder containing the .txt files
file_path = "financial_data"  # This folder should contain the .txt files

# Function to fetch file list from the local directory
def fetch_file_list():
    try:
        # Get the list of files in the 'financial_data' folder
        files = os.listdir(file_path)
        # Filter for .txt files
        txt_files = [f for f in files if f.endswith('.txt')]
        return txt_files
    except FileNotFoundError:
        st.error("The 'financial_data' folder was not found.")
        return []

# Function to read the content of a .txt file
def read_file_content(file_name):
    try:
        with open(os.path.join(file_path, file_name), "r") as file:
            content = file.read()
        return content
    except Exception as e:
        st.error(f"Error reading {file_name}: {e}")
        return ""

# Streamlit UI to display the file names and sample content
def display_files():
    file_list = fetch_file_list()
    if not file_list:
        st.write("No .txt files found in the 'financial_data' folder.")
        return

    # Display the file names and sample content
    for file_name in file_list:
        st.write(f"### File: {file_name}")  # Display the file name

        # Fetch and display the sample content of the file
        file_content = read_file_content(file_name)
        sample_data = file_content[:500]  # Displaying first 500 characters as a sample

        st.write(f"**Sample Data:**")
        st.write(sample_data)

# Run the Streamlit app
if __name__ == "__main__":
    st.title("Financial Data Files")
    display_files()
