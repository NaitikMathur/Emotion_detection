import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import scipy
from scipy import stats
from keras.models import load_model
model = load_model('/content/drive/MyDrive/emotion detection/model1.h5')
html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">AI and data Science  Master Classes </p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
  
st.title("""
        Emotion Detection
         """
         )
file= st.file_uploader("Please upload image", type=("jpg", "png"))
import cv2
from  PIL import Image, ImageOps
def import_and_predict(image):
  image_fromarray = Image.fromarray(image, 'RGB')
  resize_image = image_fromarray.resize((128, 128))
  expand_input = np.expand_dims(resize_image,axis=0)
  input_data = np.array(expand_input)
  input_data = input_data/255

  pred = loaded_model.predict(input_data)
  result = pred.argmax()
  if result == 0:
    return "Anger"
  elif result == 1:
    return "Disgust"
  elif result == 2:
    return "Fear"
  elif result == 3:
    return "Happiness"
  elif result == 4:
    return "Sadness"
  else:
    return "Surprise"
if file is None:
  st.text("Please upload an Image")
else:
  image=Image.open(file)
  #image=np.array(image)
  #file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
  #image = cv2.imdecode(file_bytes, 1)
  st.image(image,caption='Uploaded Image.', use_column_width=True)
    
if st.button("Predict"):
  result=import_and_predict(image)
  st.success('Model has predicted the image  is  of  {}'.format(result))
if st.button("About"):
  st.header("Naitik Mathur")
  st.subheader("Student, Department of Computer Engineering")
  
html_temp = """
   <div class="" style="background-color:orange;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:20px;color:white;margin-top:10px;">Image Classification Project. 14</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)


