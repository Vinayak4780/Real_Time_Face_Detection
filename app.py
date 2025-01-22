import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Streamlit app setup
st.title("Face Tracking Streamlit App")
st.text("Press 'Start' to begin webcam face tracking.")

# Load the facetracker model only once when the app starts
if 'facetracker' not in st.session_state:
    facetracker = tf.keras.models.load_model('facetracker.h5')
    st.session_state.facetracker = facetracker
else:
    facetracker = st.session_state.facetracker

# Initialize session state for controlling webcam
if "running" not in st.session_state:
    st.session_state.running = False

# Function to start the webcam feed
def webcam_face_tracking():
    cap = cv2.VideoCapture(0)
    stframe = st.empty()  # Placeholder for video frames
    
    while cap.isOpened():
        if not st.session_state.running:
            break  # Exit the loop if running is set to False
        
        _, frame = cap.read()
        frame = frame[50:500, 50:500, :]
        
        # Preprocess the frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = tf.image.resize(rgb, (120, 120))
        
        # Predict using the facetracker model
        yhat = facetracker.predict(np.expand_dims(resized / 255, 0))
        sample_coords = yhat[1][0]
        
        if yhat[0] > 0.5: 
            # Draw the main rectangle
            cv2.rectangle(frame, 
                          tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
                          tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)), 
                          (255, 0, 0), 2)
            
            # Draw the label rectangle
            cv2.rectangle(frame, 
                          tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int), [0, -30])),
                          tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int), [80, 0])), 
                          (255, 0, 0), -1)
            
            # Add text
            cv2.putText(frame, 'face', 
                        tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int), [0, -5])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Convert the frame for Streamlit display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", use_column_width=True)
    
    cap.release()
    cv2.destroyAllWindows()
    tf.keras.backend.clear_session()  # Clear the session to release memory

# Layout: Buttons for controlling webcam feed
col1, col2 = st.columns(2)

with col1:
    if st.button("Start") and not st.session_state.running:
        st.session_state.running = True
        webcam_face_tracking()

with col2:
    if st.button("End"):
        st.session_state.running = False
        st.write("Webcam feed stopped.")
        tf.keras.backend.clear_session()  # Clear session to stop model-related processes
