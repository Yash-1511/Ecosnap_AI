import streamlit as st
from PIL import Image,ImageDraw
import tensorflow as tf
import numpy as np
import cv2
class_names = [
    'GRAFFITI',
    'FADED_SIGNAGE',
    'POTHOLES',
    'GARBAGE',
    'CONSTRUCTION_ROAD',
    'BROKEN_SIGNAGE',
    'BAD_STREETLIGHT',
    'BAD_BILLBOARD',
    'SAND_ON_ROAD',
    'CLUTTER_SIDEWALK',
    'UNKEPT_FACADE'
]
st.set_page_config(layout="wide", page_title="Playground - EcoSnap")
model = tf.keras.models.load_model('models/model.h5')
st.write("## Welcom To EcoSnap Playground")
st.write(
    ":dog: Try uploading an image to watch the which type of pollutant in image. This code is open source and available [here](https://github.com/Yash-1511/DUHACKS_ECOSNAP) on GitHub :grin:"
)
st.sidebar.write("## Upload and download :gear:")


# Download the fixed image
def read_image(img):
    image = tf.image.resize(img, (224, 224))
    image= image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def fix_image(upload):
    image = np.array(Image.open(upload))
    col1.write("Original Image :camera:")
    col1.image(image)
    img = read_image(image)
    prediction = model.predict(img)
    class_name = class_names[np.argmax(prediction[0])]
    bbox = prediction[1]
    confidence = round(100 * (np.max(prediction[0])), 2)
    (xmin, xmax, ymin, ymax) = bbox[0][0] * 1920, bbox[0][1] * \
            1920, bbox[0][2] * 1080, bbox[0][3] * 1080
    bboxes = [xmin, xmax, ymin, ymax]
    st.write(class_name)
    st.write("bounding box cordinates: ",bboxes)
    col2.write("Predicted Image :wrench:")
    drawimage = cv2.rectangle(image,(int(xmax),int(ymax)),(int(xmin),int(ymin)),(255, 0, 0), 10)
    col2.image(drawimage)
    st.sidebar.markdown("\n")


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    fix_image(upload=my_upload)
else:
    st.write("please upload image")
    # fix_image("./zebra.jpg")