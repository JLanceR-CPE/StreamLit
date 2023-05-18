# -*- coding: utf-8 -*-
"""Finals Exam.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1atdD9MfGGtWhBaI3LCn-sTui_ExzlEX_
"""

#!pip install -U ipykernel

#!pip install -q streamlit



#from google.colab import drive
#drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import warnings
warnings.filterwarnings("ignore")
import os
#from matplotlib.image import imread
import random
import matplotlib.image as mpimg

import pathlib
dataset = "/content/drive/MyDrive/Colab Notebooks/RONQUILLO_Final Exam/images"

tf.random.set_seed(5)

datagen = ImageDataGenerator(rotation_range=10,
            rescale = 1./255,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False,
            zoom_range=0.1,
            shear_range=0.1,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=0.2  # set validation split to 20% 
            )

# Import the data into train and Validation subset
trainimagedata = datagen.flow_from_directory("/content/drive/MyDrive/Colab Notebooks/RONQUILLO_Final Exam/images",
                                              batch_size = 32,
                                              class_mode = 'categorical',
                                              target_size=(64,64),
                                              subset = 'training'
                                            )

testimagedata = datagen.flow_from_directory("/content/drive/MyDrive/Colab Notebooks/RONQUILLO_Final Exam/images",
                                              batch_size = 32,
                                              class_mode = 'categorical',
                                              target_size=(64,64),
                                              subset = 'validation'
                                            )

trainimagedata.classes

trainimagedata.class_indices

# Print Each sample of all Classes
# Directory path where the images are located
directory_path = '/content/drive/MyDrive/Colab Notebooks/RONQUILLO_Final Exam/images'

# List of class names
class_names = ['apple fruit', 'banana fruit', 'cherry fruit', 'chickoo fruit', 'grapes fruit', 'kiwi fruit', 'mango fruit', 'orange fruit', 'strawberry fruit']

# Create a figure to display the images
fig = plt.figure(figsize=(12, 8))

# Iterate over each class
for i, class_name in enumerate(class_names):
    # Get a list of image files in the class directory
    class_directory = os.path.join(directory_path, class_name)
    image_files = os.listdir(class_directory)

    # Select a random image file from the class
    random_image = random.choice(image_files)
    image_path = os.path.join(class_directory, random_image)

    # Load and display the image
    ax = fig.add_subplot(3, 3, i+1)
    img = mpimg.imread(image_path)
    ax.imshow(img)
    ax.set_title(class_name)
    ax.axis('off')

# Adjust the layout and display the figure
plt.tight_layout()
plt.show()

input_shape = trainimagedata.image_shape

# Model Architecture
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(64,(3,3), input_shape = input_shape,activation='relu',padding='same'))
model.add(tf.keras.layers.MaxPool2D(2,2))

model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2,2))

model.add(tf.keras.layers.Flatten())

model.add(Dense(128,activation = 'relu'))
model.add(Dropout(0.25))

model.add(Dense(64,activation = 'relu'))
model.add(Dropout(0.25))

model.add(Dense(9,activation = 'softmax'))

# Set the Hyperparameter to Adam optimizer
from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.001,beta_1=0.9,beta_2 = 0.999, epsilon=1e-8)

# Compile the model
model.compile(optimizer = optimizer,
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor = 'val_loss', patience =15)

# Fitting the model
mdl_history = model.fit(trainimagedata,
                          validation_data = testimagedata,
                          epochs=35,
                          batch_size=16,
                          callbacks=[early_stop])

tf.keras.models.save_model(model,'fruits_classifier.hdf5')

import streamlit as st
import tensorflow as tf

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('/content/flower_classifier.hdf5')
  return model
model=load_model()
st.write("""
# Plant Leaf Detection System"""
)
file=st.file_uploader("Choose plant photo from computer",type=["jpg","png"])

import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):
    size=(180,180)
    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names=['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)

import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

def load_model():
    model = tf.keras.models.load_model('/content/fruits_classifier.hdf5')
    return model

model = load_model()

st.write("""
# Plant Leaf Detection System"""
)

file = st.file_uploader("Choose plant photo from computer", type=["jpg","png"])

def import_and_predict(image_data, model):
    size = (180, 180)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names=['apple fruit', 'banana fruit', 'cherry fruit', 'chickoo fruit', 'grapes fruit', 'kiwi fruit', 'mango fruit', 'orange fruit', 'strawberry fruit']
    string = "OUTPUT : " + class_names[np.argmax(prediction)]
    st.success(string)

