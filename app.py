from __future__ import division, print_function
import cv2
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'C:\\Users\\hp\\ann_mod\\facefeatures_new_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()  # Necessary
face_classifier = cv2.CascadeClassifier('C:\\Users\\hp\\Downloads\\haarcascade_frontalface_default.xml')
print('Model loaded. Check http://127.0.0.1:5000/')

nm_l = []
main_dir = "C:\\Users\\hp\\train"
for root, dirs, files in os.walk(main_dir):
    for i in dirs:
        nm_l.append(i)


def model_predict(img_path, model):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        img_r = cv2.resize(roi_color, (224, 224))
        img_exp = np.expand_dims(img_r, axis=0)
        x = preprocess_input(img_exp, mode='caffe')

        preds = model.predict(x)
        return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)

        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path, model)
        ind = np.argmax(result)
        result = nm_l[ind]
        print(result)
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

