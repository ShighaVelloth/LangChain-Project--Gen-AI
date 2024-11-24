import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from gtts import gTTS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import os

# Set the path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Set up Google API Key
with open("C:/Gen AI APP/Langachain Project/Keys.txt") as f:
    GOOGLE_API_KEY = f.read().strip()

# Initialize LangChain with Google Generative AI model
chat_model = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-1.5-flash")

# Define Chat Prompt Template
chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant designed for visually impaired individuals. "
               "Provide a detailed description of the image, including objects, text, symbols, traffic signs, hazards, or any relevant contextual information. "
               "Highlight detected text and include it in the description. Indicate potential dangers and give a clear explanation of the scene."),
    ("human", "Please describe this image: {image_description}")
])

# Initialize output parser
output_parser = StrOutputParser()

# Create the full chain (Prompt -> Model -> Output Parser)
chain = chat_prompt_template | chat_model | output_parser

# Set up Streamlit tabs
tabs = st.tabs(["Home", "Upload Image", "About"])

# --- Home Tab ---
with tabs[0]:
    st.title("Welcome to the AI Assistant!")
    st.write(
        """
        This AI assistant is designed to help visually impaired individuals understand their surroundings through:
        - Detailed scene descriptions.
        - Text detection and speech synthesis.
        - Hazard and warning detection in images.
        """
    )
    st.markdown("### Features")
    st.success("1. **Image-to-Text and Audio Conversion**")
    st.info("2. **Contextual Scene Understanding**")
    st.warning("3. **Hazard Detection and Safety Alerts**")

# --- Upload Image Tab ---
with tabs[1]:
    st.title("Upload an Image to Analyze")

    # Image uploader
    uploaded_image = st.file_uploader("Choose an image (JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        try:
            # Display the uploaded image
            st.subheader("Uploaded Image:")
            st.image(uploaded_image, caption="Uploaded Image Preview", use_container_width=True)

            # Load and process the image
            image = Image.open(uploaded_image)

            # Enhance the image for better OCR performance
            image = image.convert("L")  # Convert to grayscale
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2)  # Increase contrast
            image = image.filter(ImageFilter.SHARPEN)  # Sharpen the image

            # Extract text using Tesseract
            extracted_text = pytesseract.image_to_string(image)
            st.subheader("Extracted Text:")
            st.write(extracted_text if extracted_text.strip() else "No text detected.")

            # Generate description using LangChain
            image_description = extracted_text if extracted_text.strip() else "No text detected. Describe the objects and context of the image."
            user_input = {"image_description": image_description}

            description = chain.invoke(user_input)
            st.subheader("AI-Generated Description:")
            st.write(description)

            # Convert description and text to audio
            st.subheader("Audio Description:")
            full_description = f"{description}\n{extracted_text}" if extracted_text.strip() else description
            speech = gTTS(text=full_description, lang='en', slow=False)
            audio_file = "audio_description.mp3"
            speech.save(audio_file)

            # Provide audio playback
            with open(audio_file, "rb") as audio:
                st.audio(audio.read(), format="audio/mp3")
            os.remove(audio_file)  # Clean up the audio file after playback

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("Please upload an image to begin.")

# --- About Tab ---
with tabs[2]:
    st.title("About This App")
    st.write(
        """
        This application leverages cutting-edge AI technologies to make vision accessible for visually impaired users.
        It uses:
        - **Tesseract OCR** for text extraction.
        - **Google Generative AI (Gemini)** for scene understanding.
        - **gTTS** for text-to-speech conversion.
        
        Developed with a mission to enhance accessibility and provide real-time insights into the visual world.
        """
    )
    st.markdown("**Technologies Used:** Python, Streamlit, Tesseract, LangChain, Google Generative AI.")
