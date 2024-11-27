import streamlit as st
import google.generativeai as genai

# Configure the Generative AI API
f = open("C:/Gen AI APP/Keys.txt")
key = f.read()
genai.configure(api_key=key)

# Sidebar Sections
st.sidebar.title("AI Code Reviewer")

# About Section
st.sidebar.header("About")
st.sidebar.info(
    """
    The **AI Code Reviewer** is designed to help developers analyze, debug, and improve their code using advanced AI technology.
    - Quickly identify bugs and errors.
    - Receive helpful suggestions for optimization.
    - Improve your coding efficiency and skills.
    """
)

# Steps Section
st.sidebar.header("Steps")
st.sidebar.markdown(
    """
    1. Enter your code in the input box.
    2. Click the **Review** button.
    3. View AI-generated feedback and suggestions.
    """
)

# Thank You Section
st.sidebar.header("Thank You")
st.sidebar.markdown(
    """
    Thank you for using the AI Code Reviewer! Your feedback and suggestions are invaluable to improving this tool.  
    Happy coding! 🎉
    """
)

# Set system instructions
system_instruction = "You are a coder. Review code written by humans and provide the correct code."

# Define the generative model
model = genai.GenerativeModel(model_name='gemini-1.5-flash')

# Streamlit App Title 
st.markdown(
    """
    <h1 style="text-align: center; color: #4CAF50;">AI Code Reviewer</h1>
    <p style="text-align: center; font-size: 18px;">Your personal AI assistant for reviewing code</p>
    """,
    unsafe_allow_html=True,
)

# Input Section
st.markdown("### Enter Your Code Below:")
user_prompt = st.text_area("Code Input:", placeholder="Type your code here...", key="user_code_input")

# Button 
btn_click = st.button("Review", key="review_button")


if btn_click:
    if user_prompt.strip():
        try:
            
            response = model.generate_content(user_prompt)
            ai_response = response.text.strip()

            st.markdown("### AI Review:")
            st.code(ai_response)
        except Exception as e:
            st.error(f"An error occurred while processing your request: {e}")
    else:
        st.warning("Please enter some code to review.")

# Footer Design
st.markdown(
    """
    <hr>
    <p style="text-align: center; font-size: 12px; color: #555;">
        Powered by Streamlit and Google AI
    </p>
    """,
    unsafe_allow_html=True,
)
