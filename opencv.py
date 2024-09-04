import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('recycle_classifier2.keras')
class_labels = ['Card Board', 'Compost', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
Recycle = ['Paper','Plastic','Glass','Card Board','Metal']
non_Recycle = ['Trash', 'Compost']

def preprocess_frame(frame):
    img = cv2.resize(frame, (64, 64)) 
    img_array = np.expand_dims(img, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array

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

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() 
    if not ret:
        break
    
    label = predict_frame_class(frame)

    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Real-Time Waste Classification', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
