
from flask import Flask, render_template, request, url_for, redirect, session, jsonify
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import numpy as np
import traceback
from datetime import datetime
import os
from PIL import Image

# Load your trained model
model=load_model('model_inception.h5')

# Define a flask app
app = Flask(__name__)



def model_prediction(input, model):

    input_img = load_img(input, target_size = (224,224))

    # Preprocessing the image
    input_img= img_to_array(input_img)
    input_img = np.expand_dims(input_img, axis=0)
    input_img = preprocess_input(input_img)
    preds = np.argmax(model.predict(input_img), axis = 1)

    if preds==0:
        result="The Disease is Tomato___Bacterial_spot"
    elif preds==1:
        result="The Disease is Early_blight"
    elif preds==2:
        result="The Disease is healthy"
    elif preds==3:
        result="Te Disease is Late_blight"
    elif preds==4:
        result="The Disease is Leaf_Mold"
    elif preds==5:
        result="The Disease is Septoria_leaf_spot"
    elif preds==6:
        result="The Disease is Spider_mites Two-spotted_spider_mite"
    elif preds==7:
        result="The Disease is Target_Spot"
    elif preds==8:
        result="The Disease is Tomato_mosaic_virus"
    elif preds==9:
        result="The Disease is Tomato_Yellow_Leaf_Curl_Virus"

    return result

upload_folder = os.path.join('static', 'uploads')
 
app.config['UPLOAD'] = upload_folder


@app.route('/', methods=["GET", "POST"])
def index():
    # Main page
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if model:
        try:
             # Get the file from post request
            if request.method == 'POST':                      
            # Get the file from post request
                f = request.files['file']

            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
           
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))

            f.save(file_path)
            filess = '/uploads' + '/'+secure_filename(f.filename)
            print(filess)
            # image_names = os.listdir('/Users/VA20463476/Desktop/projects/Leaf_project/image/uploads')
            
            # Make prediction
            prediction = model_prediction(file_path, model)
            return render_template('index.html', Pred= prediction, img = filess)

        except:
            return jsonify({"trace ": traceback.format_exc()})

    else:
        return ("no model is here to use")


if __name__ == '__main__':
    app.run(port=5001,debug=True)
