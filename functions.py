import pymysql
import cv2
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import requests
# Create a connection object
"""
conn = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='',
    db='eventasya'
)
def extract_images(conn,model):
    images=[]
    target_size=(299, 299)
    cursor = conn.cursor()
    # Execute a SQL query to select the venue ID and image fields
    cursor.execute("SELECT venue_id, image FROM venues_venueimages")
    # Fetch the results
    results = cursor.fetchall()
    conn.commit()
    cursor.close()
    for result in results:
        # Get the venue ID and image path from the result set
        venue_id = result[0]
        path = result[1]
        # Download the image data
        img = requests.get(path)
        img_data = img.content
        # Decode the image data into a NumPy array
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # Convert the image to a NumPy array
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, target_size)
        images.append(img_resized)
        # Add the ID and image to the dictionary
    imgs=preprocess_dataset(images, model)
    # Create a dictionary to store the ID and preprocessed image for each row
    id_image_dict = {}
    for i in range(len(imgs)):
        # Get the venue ID from the corresponding result
        venue_id = results[i][0]
        # Add the ID and preprocessed image to the dictionary
        id_image_dict[venue_id] = imgs[i]
    return id_image_dict
# Preprocess dataset
"""
def preprocess_request(file):
    target_size = (299, 299)
    img = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, target_size)
    img = img.reshape((1, 299, 299, 3))

    return img
def preprocess_dataset(x, model):
    X_feats = []
    for test in x:
        if test is None:
            print('none')
            continue  # Skip iteration if test is None
        try:
            Query_image = test
            Query_image = preprocess_input(Query_image)
            Query_feats = model.predict(np.expand_dims(Query_image, axis=0))
            Query_feats = Query_feats.squeeze()
            X_feats.append(Query_feats)
        except Exception as e:
            print(f'Error preprocessing image: {e}')
    return np.array(X_feats)

# Test model
def similar_img(user_input,db_imgs):
  # Euclidean distance
    results = []
    for key , value in db_imgs.items():
        try:
            d = np.linalg.norm(value.flatten() - user_input.flatten())
            results.append((d, key))
        except:
              print('error',key)
    results = sorted(results)
    return results
def display_id(result):
    ids=[]
    for d,k in result:
        if (d< 0.5):
            ids.append(k)
    return ids
