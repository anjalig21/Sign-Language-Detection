import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np

model = tf.keras.models.load_model('anjalis-model.h5')

frameWidth = 640
frameHeight = 480

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

while True:
    _,image = cap.read(0)
    imgWebcam = cv2.flip(image, 1)

    img = cv2.cvtColor(imgWebcam, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (32, 32))
    img = np.expand_dims(img, 0)
    predictions = model.predict(img)
    predictions = np.argmax(predictions)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(imgWebcam, chr(predictions + 65), (10, 100), font, 3, (0, 255, 0), 10)

    cv2.imshow("Window", imgWebcam)
    cv2.waitKey(1)


