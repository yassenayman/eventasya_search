from flask import Flask, jsonify, request
from keras.models import load_model
import functions
import cv2
import numpy as np
import schedule
import time
import threading
app = Flask(__name__)

model = load_model('B:/fine_tuned_model.h5')
lock = threading.Lock()
def update():
    global imgs
    imgs = functions.extract_images(functions.conn, model)

schedule.every(10).seconds.do(update)

while True:
    schedule.run_pending()
    time.sleep(1)
# Start the scheduled job before running the Flask app
scheduled_job_thread = threading.Thread(target=scheduled_job)
scheduled_job_thread.daemon = True
scheduled_job_thread.start()

# Define a route for making predictions
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
