import numpy as np
import os
import pickle
import streamlit as st
from PIL import Image
import sys
import tensorflow as tf
import urllib
import cv2
import argparse
from io import StringIO
import pickle

from facerec.utils import load_image, get_face_data, to_bgr, recognize_face, annotate_image


st.text('Upload photos to recognize faces!')

FILE_TYPES = ['.jpg', '.png']

with open('encodings.pickle', 'rb') as handle:
    database = pickle.load(handle)

parser = argparse.ArgumentParser()
parser.add_argument('--path', help='Specify path to meta data')

uploaded_file = st.file_uploader('Choose file')

show_file = st.empty()
if not uploaded_file:
    show_file.info("Please upload a file of type: " + ", ".join(FILE_TYPES))

result_file = st.empty()
if not result_file:
    result_file.info("Output: ")

if uploaded_file:
    # img_bytes_io = uploaded_file.copy()
    img_bytes = uploaded_file.read()

    # show_file.image(img_bytes)

    image_rgb = np.array(Image.open(uploaded_file).resize((600, 400), Image.ANTIALIAS))

    boxes, encodings = get_face_data(image_rgb, 'cnn')

    names = []

    for encoding in encodings:
        name = recognize_face(encoding, database)
        names.append(name)

    # put bounding boxes and names around the faces
    image = annotate_image(image_rgb, boxes, names)

    if image is not None:
        show_file.image(image)
