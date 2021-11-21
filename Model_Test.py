import cv2 as cv
import numpy as np
from numpy.lib.function_base import disp
import tensorflow as tf
import matplotlib.pyplot as plt

cap = cv.VideoCapture(0)
model_path = r'./checkpoints/age_range_model1.epoch16-loss4.51.hdf5'
model = tf.keras.models.load_model(model_path)

face_cascade = cv.CascadeClassifier('./cascade/haarcascade_frontalface.xml');

is_age_range = "age_range" in model_path

count = 0
sum = 0

while True: 
    ret, frame = cap.read()
    sq_img = np.array([row[:480] for row in np.asarray(frame)])
    gray = cv.cvtColor(sq_img, cv.COLOR_BGR2GRAY)
    small = cv.resize(gray, (48, 48))

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv.rectangle(sq_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    img = np.asarray(small).reshape((48,48,1))
    pred = model.predict(np.array([img]))[0][0] 

    display_text = ""

    if pred != 0:
        count += 1
        if is_age_range:
            sum += 5 * pred
            display_text = "predicted age range is: " + str(round(pred * 5)) + " to " + str((round(pred + 1) * 5))
            print(display_text)
        else: 
            sum += pred
            display_text = "Predicted age: " + str(round(pred))
            print(display_text)

    if (len(faces) >= 1):
        sq_img = cv.putText(sq_img, display_text, faces[0][:2], cv.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)

    cv.imshow('frame', sq_img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

print("Avg age: ", sum // count)