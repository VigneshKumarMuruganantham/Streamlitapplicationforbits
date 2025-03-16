import streamlit as st

# Define global variables
global user_input, output_1, output_2
user_input = ""
output_1 = "Vaa da vaa da"
output_2 = "Vaa da vaa da"

# Function to print user input in the Python console
def print_to_console(input_value):
    print(f"User Input in Console: {input_value}")

# Function to update output fields based on input
def update_outputs(input_value):
    global output_1, output_2
    # You can write your logic here to update the output fields
    output_1 = f"Processed Output 1 for: {input_value}"
    output_2 = f"Processed Output 2 for: {input_value}"

# Streamlit app
def main():
    global user_input, output_1, output_2  # Access the global variables

    st.title("User Prompt Input with Outputs")

    # Input box to get prompt from the user
    user_input = st.text_input("Enter a prompt:")

    # Button to submit the input
    if st.button("Submit"):
        # Display the user input
        st.write(f"You entered: {user_input}")
        
        # Call the function to print input to the Python console
        print_to_console(user_input)
        
        # Call the function to update the output fields based on user input
        update_outputs(user_input)

    # Display the two output fields
    st.text_area("Output Field 1", value=output_1, height=100)
    st.text_area("Output Field 2", value=output_2, height=100)

if __name__ == "__main__":
    main()
