from flask import Flask, jsonify, request
from keras.models import load_model
import functions
import cv2
import numpy as np
import schedule
import time
import threading
import streamlit as st
import zipfile
app = Flask(__name__)

# Define the path to your zipped file and the path to extract the contents to
zip_path = 'fine_tuned_model.zip'
extract_path = 'fine_tuned_model'

# Unzip the file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Load the model from the extracted files
model_path = f'{extract_path}/fine_tuned_model.h5'
model = load_model(model_path)

"""lock = threading.Lock()
def update():
    global imgs
    imgs = functions.extract_images(functions.conn, model)

#schedule.every(10).seconds.do(update)

while True:
    schedule.run_pending()
    time.sleep(1)
# Start the scheduled job before running the Flask app
scheduled_job_thread = threading.Thread(target=scheduled_job)
scheduled_job_thread.daemon = True
scheduled_job_thread.start()
"""
# Define a route for making predictions
global imgs
imgs={}
imgs={1:[0.00830947, 0.10077339, 0.47513428, 0.41578284], 10: [0.02004314, 0.24340728, 0.5684739 , 0.1680757 ] ,789: [0.01972592, 0.09381499, 0.6248155 , 0.26164353]}

@app.route('/predict', methods=['POST'])
def predict():
    # Load the image file from the request
    file = request.files['image'].read()
    img=functions.preprocess_request(file)
    # Read the file contents into a NumPy array
    image=functions.preprocess_dataset(img,model)
    print ('done')
    result = functions.similar_img(image, imgs)
    ids = functions.display_id(result)
    # Return the predicted label as JSON
    return jsonify({'venue_ids': ids})


if __name__ == '__main__':
    app.run(debug=True)
