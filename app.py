from flask import Flask, render_template, request, redirect,  url_for
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import vgg16
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras import models

import numpy as np
import pandas as pd
import cv2, os
import imutils

app = Flask(__name__, template_folder='templates')

UPLOAD_FOLDER = "./static"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return (render_template('index.html'))
    if request.method == 'POST':
            f = request.files['file']
            print(f.filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))

            img = cv2.imread('static/'+ f.filename)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgRGB2 = cv2.resize(imgRGB, (48, 48))
            imgRGB2 = imgRGB2.astype("float") / 255.0
            imgRGB2 = img_to_array(imgRGB2)
            imgRGB2 = np.expand_dims(imgRGB2, axis=0)

            df = pd.read_csv('bird_species.csv')
            #df['speciesname']

            IMAGE_DIMS = (48, 48, 3)
            base_model = vgg16.VGG16(weights='imagenet',
                                    include_top=False,
                                   input_shape=IMAGE_DIMS)
            # Freese base layers
            for layer in base_model.layers:
                layer.trainable = False
            # Build layers
            # creating own layers in addition to VGG16 as the base layers
            model = models.Sequential(base_model.layers)
            model.add(Flatten())
            model.add(Dense(4096, activation='relu'))
            model.add(Dense(33, activation='softmax'))
            print(model.summary())

            model.load_weights('model/birdsCNN_best.h5')
            probs = model.predict(imgRGB2)

            print(probs)
            idx1 = np.argmax(probs)
            myresults = df['speciesname'][idx1]
    return (render_template('index.html',mybird=f.filename, result=myresults))

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True,port="5000")