import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from PIL import Image

# Load the model architecture from the JSON file
with open('model.json', 'r') as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)

# Load the model weights from the H5 file
model.load_weights('model.h5')

# Define the class labels
class_labels = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Preprocess the frame for the model
def preprocess_frame(frame, face_coords):
    (x, y, w, h) = face_coords
    face = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray, (48, 48))
    normalized_frame = resized_frame / 255.0
    reshaped_frame = np.reshape(normalized_frame, (1, 48, 48, 1))
    return reshaped_frame

# Detect faces and draw bounding boxes
def detect_faces_and_predict(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        face_coords = (x, y, w, h)
        preprocessed_frame = preprocess_frame(frame, face_coords)
        
        # Predict the expression
        predictions = model.predict(preprocessed_frame)
        max_index = np.argmax(predictions[0])
        predicted_label = class_labels[max_index]
        
        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    return frame

# Streamlit UI
st.set_page_config(page_title="Facial Expression Detection", page_icon="😊", layout="wide")

st.title("Real Time Mood Detection Using Deep Learning and Streamlit Integration")
st.write("This app can detect facial expressions in real-time using your webcam or from an uploaded image.")

# Sidebar with options
st.sidebar.header("Options")
start_webcam = st.sidebar.button("Start Webcam", key="start_webcam_button")
stop_webcam = st.sidebar.button("Stop Webcam", key="stop_webcam_button")

# Image file upload
uploaded_image = st.sidebar.file_uploader("Upload an image for facial expression detection", type=["jpg", "jpeg", "png"])

# Placeholder for webcam feed
webcam_placeholder = st.empty()

# Initialize session state
if "webcam_active" not in st.session_state:
    st.session_state.webcam_active = False

# Function to release the webcam
def release_webcam():
    if "cap" in st.session_state and st.session_state.cap.isOpened():
        st.session_state.cap.release()

if start_webcam:
    st.session_state.webcam_active = True
    st.session_state.cap = cv2.VideoCapture(0)
    if not st.session_state.cap.isOpened():
        st.error("Error: Could not open webcam.")
        st.session_state.webcam_active = False
    else:
        st.success("Webcam opened successfully.")

if stop_webcam:
    st.session_state.webcam_active = False
    release_webcam()
    st.success("Webcam stopped.")

while st.session_state.webcam_active:
    ret, frame = st.session_state.cap.read()
    if not ret:
        st.error("Error: Could not read frame.")
        break

    # Detect faces and predict expressions
    frame = detect_faces_and_predict(frame)
    
    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Display the frame in Streamlit
    webcam_placeholder.image(frame_rgb, channels='RGB')

# Process uploaded image
if uploaded_image is not None:
    st.subheader("Uploaded Image")
    image = Image.open(uploaded_image)
    frame = np.array(image)
    
    # Detect faces and predict expressions
    frame = detect_faces_and_predict(frame)
    
    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Display the frame in Streamlit
    frame_image = Image.fromarray(frame_rgb)
    st.image(frame_image, caption='Uploaded Image', use_column_width=True)

# Footer with project credits
st.markdown("---")
st.markdown("### Project by: Muhammad Shoaib Arshad & Rehman Anwar")
st.markdown("### Supervised by: Mr Asif")
