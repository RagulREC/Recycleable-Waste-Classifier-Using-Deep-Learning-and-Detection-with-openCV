import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tkinter import Tk, filedialog
from PIL import Image

model = tf.keras.models.load_model('recycle_classifier2.keras')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64)) 
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0 
    return img_array

def predict_image_class(img_path):
    img_array = preprocess_image(img_path)
    result = model.predict(img_array)
    predicted_class = np.argmax(result, axis=1)

    class_labels = ['Card Board', 'Compost', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash'] 
    Recycle = ['Paper','Plastic','Glass','Card Board','Metal']
    non_Recycle = ['Trash', 'Compost']
    prediction = class_labels[predicted_class[0]]
    
    print(f'The image is predicted to be: {prediction}')
    if prediction in Recycle:
        print(f'The image is predicted to be Recycle waste')
    elif prediction in non_Recycle:
        print(f'The image is predicted to be Non-Recycle Waste')

def upload_image():
    root = Tk()
    root.withdraw() 
    img_path = filedialog.askopenfilename() 
    if img_path:
        print(f"Selected Image: {img_path}")
        predict_image_class(img_path)
    else:
        print("No image selected!")

if __name__ == "__main__":
    upload_image()
