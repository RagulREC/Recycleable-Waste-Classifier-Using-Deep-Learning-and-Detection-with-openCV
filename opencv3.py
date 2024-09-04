import cv2
import tensorflow as tf
import numpy as np
import pygame as mixer
import os
from pygame.mixer import Sound
mixer.init()
sound =Sound('alarm.wav')

# Load the trained model
model = tf.keras.models.load_model('recycle_classifier2.keras')

# Define class labels
class_labels = ['Card Board', 'Compost', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
Recycle = ['Paper','Plastic','Glass','Card Board','Metal']
non_Recycle = ['Trash', 'Compost']

# Function to preprocess the ROI (Region of Interest)
def preprocess_frame(frame):
    img = cv2.resize(frame, (64, 64))  # Resize the image to match the input shape of the model
    img_array = np.expand_dims(img, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Function to predict the class of the frame
def predict_frame_class(frame):
    img_array = preprocess_frame(frame)
    result = model.predict(img_array)
    predicted_class = np.argmax(result, axis=1)
    prediction = class_labels[predicted_class[0]]

    if prediction in Recycle:
        label = f'{prediction} (Recycle)'
    else:
        label = f'{prediction} (Non-Recycle)'

    return label

def music_play(frame):
    img_array = preprocess_frame(frame)
    result = model.predict(img_array)
    predicted_class = np.argmax(result, axis=1)
    prediction = class_labels[predicted_class[0]]

    if prediction in Recycle:
        labl = True
    else:
        labl = False

    return labl

# Start video capture using OpenCV
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break
    
    # Define the Region of Interest (ROI)
    height, width, _ = frame.shape
    roi_x1, roi_y1 = int(width * 0.3), int(height * 0.3)  # Top-left corner
    roi_x2, roi_y2 = int(width * 0.7), int(height * 0.7)  # Bottom-right corner
    
    # Extract the ROI from the frame
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    
    # Predict the class of the ROI
    label = predict_frame_class(roi)
    decision = music_play(roi)
    if not decision:
        sound.play()


    
    # Draw the rectangle (ROI) on the frame
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)  # Blue rectangle
    
    # Display the label on the frame
    cv2.putText(frame, label, (roi_x1, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Show the frame with prediction
    cv2.imshow('Real-Time Waste Classification', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()