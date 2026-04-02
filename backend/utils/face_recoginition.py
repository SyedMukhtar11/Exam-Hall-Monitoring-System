import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model2.h5")

names = ["student1","student2","student3","student4"]

def recognize_face(img):

    img = cv2.resize(img,(64,64))
    img = img/255.0
    img = np.reshape(img,(1,64,64,3))

    pred = model.predict(img,verbose=0)

    return names[np.argmax(pred)]