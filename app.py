import streamlit as st
from PIL import Image
import easyocr
from gtts import gTTS
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from collections import Counter
import numpy as np
import os

# Load Google API Key
with open("C:/Gen AI APP/Langachain Project/Keys.txt") as f:
    GOOGLE_API_KEY = f.read().strip()

# Initialize LangChain with Google Generative AI model
chat_model = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-1.5-flash")

# Initialize EasyOCR
ocr_reader = easyocr.Reader(['en'], gpu=False)

# Initialize YOLO
yolo_model = YOLO("C:/Gen AI APP/Langachain Project/yolov3.pt")

# Initialize BLIP
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Streamlit Page Setup
st.set_page_config(page_title="AI Vision Assistant", layout="wide")
st.title("Eyes of AI: Empowering the Visually Impaired with Smarter Image Analysis")
st.markdown("""
Upload an image for:
- *Scene Description*: A concise description of the image's content and context.
- *Extracted Text*: Text from the image.
- *Detected Objects*: A list of identified objects in the image.
- *AI Assistance*: Detailed summary of the image.
- *Audio Output*: Listen to the results.
""")

# File Uploader
uploaded_image = st.file_uploader("Upload an Image:", type=["jpg", "jpeg", "png"])

# Utility functions
def extract_text(image):
    """Extract text using EasyOCR."""
    results = ocr_reader.readtext(np.array(image), detail=0)
    return " ".join(results) if results else "No text detected."

def detect_objects(image):
    """Detect objects using YOLO and group them with counts."""
    # Ensure the image has 3 channels (convert RGBA to RGB if necessary)
    if image.mode == "RGBA":
        image = image.convert("RGB")
    
    # Convert PIL image to NumPy array
    image_np = np.array(image)
    
    # Run YOLO object detection
    results = yolo_model(image_np)
    
    # Extract detected object labels
    detected_labels = [yolo_model.names[int(box.cls)] for box in results[0].boxes]
    
    # Count occurrences of each detected object
    object_counts = Counter(detected_labels)
    
    # Format the output as "count object_name(s)"
    formatted_objects = [
        f"{count} {label}{'s' if count > 1 else ''}" for label, count in object_counts.items()
    ]
    
    return ", ".join(formatted_objects) if formatted_objects else "No objects detected."

def describe_scene(image):
    """Generate a scene description using the BLIP model."""
    inputs = blip_processor(image, return_tensors="pt")
    output = blip_model.generate(**inputs)
    return blip_processor.decode(output[0], skip_special_tokens=True)

def generate_response(objects, text, scene_description):
    """Generate AI response using LangChain."""
    base_prompt = f'''
    I am a visually impaired individual. I need assistance in understanding my surroundings based on the image I uploaded.
    Please describe the scene in detail, considering the following:

    - Key objects in the scene
    - Any text or labels in the image
    - How the objects are arranged
    - Any potential safety hazards or obstacles

    Scene Description: {scene_description}
    Objects detected: {objects}
    Text extracted: {text}
    '''
    
    prompt = ChatPromptTemplate.from_messages([("system", base_prompt), ("user", base_prompt)])
    output_parser = StrOutputParser()
    
    # Generate the response using LangChain's chain method
    chain = prompt | chat_model | output_parser
    try:
        result = chain.invoke({})
        return result
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, something went wrong. Please try again."

def text_to_audio(text):
    """Convert text to speech."""
    audio_file = "output.mp3"
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(audio_file)
    return audio_file

# Main app logic
if uploaded_image:
    # Layout Configuration
    col1, col2 = st.columns([1, 2])

    # Display Uploaded Image in the Left Column
    with col1:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Process Image and Display Results in the Right Column
    with col2:
        with st.spinner("Processing image..."):
            # Extract text and detect objects
            text = extract_text(image)
            objects = detect_objects(image)
            
            # Generate Scene Description using BLIP
            scene_description = describe_scene(image)
            
            # Generate AI Response
            ai_response = generate_response(objects, text, scene_description)

        # Scene Description
        st.subheader("Scene Description")
        st.write(scene_description)

        # Extracted Text
        st.subheader("Extracted Text")
        st.write(text)

        # Detected Objects
        st.subheader("Detected Objects")
        st.write(objects if objects else "No objects detected.")

        # AI Response
        st.subheader("AI Assistance")
        st.write(ai_response)

        # Audio Summary
        st.subheader("Audio")
        with st.spinner("Generating audio..."):
            combined_text = f"{scene_description}\nExtracted Text: {text}\nAI Assistance: {ai_response}"
            audio_file = text_to_audio(combined_text)
            with open(audio_file, "rb") as file:
                st.audio(file.read(), format="audio/mp3")
            os.remove(audio_file)
else:
    st.info("Upload an image to see results.")
