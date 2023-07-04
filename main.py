from flask import Flask, jsonify, request
from keras.models import load_model
import functions
import cv2
import numpy as np
import schedule
import time
import threading
import zipfile

app = Flask(__name__)

# Define the path to your zipped file and the path to extract the contents to
zip_path = 'https://raw.githubusercontent.com/yassenayman/eventasya_search/main/fine_tuned_model.zip'
# Download the contents of the ZIP file
response = requests.get(zip_path)

# Extract the contents of the ZIP file to a directory
with zipfile.ZipFile(io.BytesIO(response.content)).extractall('fine_tuned_model')

# Load the model from the extracted files
model_path = 'fine_tuned_model/fine_tuned_model.h5'
model = load_model(model_path)


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


# Start the Flask application server
if __name__ == '__main__':
    app.run()
