import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('fruits_classifier.hdf5')
    return model

model = load_model()

st.write("""
# Fruit Detection System"""
)

file = st.file_uploader("Choose a fruit photo",type=["jpg","png"])

def import_and_predict(image_data, model):
    size = (64, 64) 
    
    # Resize the image to the expected input shape of the model
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)

    # Reshape and normalize the image
    img_reshape = img[np.newaxis,...]
    img_reshape = img_reshape/255 # Assuming your model was trained on images normalized in the range [0,1]

    # Make predictions using the Keras model
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['apple fruit', 'banana fruit', 'cherry fruit', 'chickoo fruit', 'grapes fruit', 'kiwi fruit', 'mango fruit', 'orange fruit', 'strawberry fruit']
    string = "OUTPUT : " + class_names[np.argmax(prediction)]
    st.success(string)
