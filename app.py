import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import gradio as gr

# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Load pre-trained model
model = tf.keras.models.load_model('emotion_detector_model.h5')

# Image preprocessing function
def preprocess_image(image):
    # Resize image to 48x48
    face = cv2.resize(image, (48, 48))
    # Convert image to grayscale if it's not already
    if len(face.shape) == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = np.expand_dims(face, axis=-1)  # Add channel dimension
    face = np.expand_dims(face, axis=0)   # Add batch dimension
    face = face / 255.0  # Normalize pixel values
    return face

# Emotion detection function
def detect_emotion(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(processed_image)
    
    # Get the emotion with the highest confidence
    emotion_idx = np.argmax(prediction[0])
    emotion = emotion_labels[emotion_idx]
    
    return emotion

# Gradio interface
gr.Interface(fn=detect_emotion, inputs="image", outputs="text", title="Emotion Detector").launch(share=True)
