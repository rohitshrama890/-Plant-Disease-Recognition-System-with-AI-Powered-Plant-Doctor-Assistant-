import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load TensorFlow Model
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Get Chatbot Response using GPT-4
def get_chatbot_response(disease, question):
    try:
        api_key = os.getenv("OPENAI_API_KEY")  # Fetch API key from environment variable
        if not api_key:
            st.error("API key for OpenAI not found. Please set it as an environment variable.")
            return "Error: API key not found."

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"The identified plant disease is {disease}. The user asks: {question}. How should they cure it?"}
        ]
        data = {
            "model": "gpt-4",
            "messages": messages,
            "max_tokens": 100
        }
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            data=json.dumps(data)
        )

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return "Error: Unable to get a response from the chatbot."

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return "Error: Unable to get a response from the chatbot."

# Initialize session state for disease prediction and chatbot response
if 'predicted_disease' not in st.session_state:
    st.session_state.predicted_disease = None
if 'chatbot_response' not in st.session_state:
    st.session_state.chatbot_response = None

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Disease Recognition And Chatbot Assistant"])

# Main Page
if app_mode == "Home":
    st.header("Plant Disease Recognition System and AI-Powered Plant Doctor Assistant🌿")
    st.markdown("""
    
    Welcome!! 

    Our mission is to help in identifying plant diseases efficiently and provide expert advice on curing them. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. With the AI-powered Plant Doctor Assistant, you can also ask questions about the disease and receive solutions for its cure. Together, let's protect our crops and ensure a healthier harvest!
    
    ### How It Works

    1. **Upload Image:** Go to the Disease Recognition page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.
    4. **AI-Powered Plant Doctor Assistant:** Ask questions about the diagnosed disease and receive expert guidance on how to cure it.

    """)

# Prediction Page
elif app_mode == "Disease Recognition And Chatbot Assistant":
    st.header("Disease Recognition And Chatbot Assistant")
    
    

    test_image = st.file_uploader("Choose an Image:")
    
    from PIL import Image

    if test_image:
    # Open the uploaded image file
        image = Image.open(test_image)
    
    # Resize the image (e.g., 200x200 pixels)
        image = image.resize((200, 200))

    # Display the resized image
        st.image(image, use_column_width=False)

    
    if st.button("Predict"):
        if test_image:
            st.snow()
            st.write("Our Prediction")
            result_index = model_prediction(test_image)

            # Reading Labels
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy']
            
            st.session_state.predicted_disease = class_name[result_index]  # Store prediction in session state
            st.success(f"Model is Predicting it's a {st.session_state.predicted_disease}")

    # Display the chatbot interaction section if a prediction was made
    if st.session_state.predicted_disease:
        question = st.text_input("Ask a question about the disease")

        if st.button("Get Chatbot Answer"):
            if question:
                st.session_state.chatbot_response = get_chatbot_response(st.session_state.predicted_disease, question)  # Store chatbot response in session state
                st.write(f"Docter Assistant: {st.session_state.chatbot_response}")
            else:
                st.write("Please ask a question.")
