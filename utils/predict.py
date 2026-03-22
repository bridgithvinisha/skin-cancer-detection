import numpy as np
import cv2
from tensorflow.keras.models import load_model

IMG_SIZE = 224

model = load_model("model/skin_cancer_model.keras")
classes = np.load("model/classes.npy", allow_pickle=True)

def predict_image(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return classes[class_index], confidence