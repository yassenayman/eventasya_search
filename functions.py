import pymysql
import cv2
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import requests
# Create a connection object

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
