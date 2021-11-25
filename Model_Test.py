import cv2 as cv
import numpy as np
from numpy.lib.function_base import disp
import tensorflow as tf
import matplotlib.pyplot as plt

cap = cv.VideoCapture(0)
model_path = r'./checkpoints/age_range_model2_classification_relu.epoch16-loss1.78.hdf5'
model = tf.keras.models.load_model(model_path)

face_cascade = cv.CascadeClassifier('./cascade/haarcascade_frontalface.xml');

is_age_range = "age_range" in model_path
is_classification = "classification" in model_path

count = 0
sum = 0

while True: 
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    display_text = ""

    if (len(faces) >= 1):
        (x, y, w, h) = faces[0]
        cropped = np.array([row[x + 30: x + w - 30] for row in gray[y + 5: y + h - 5]])
        small = cv.resize(cropped, (48, 48))

        img = np.asarray(small).reshape((48,48,1))
        cv.imshow('small ', img)

        pred = model.predict(np.array([img]))[0]

        if is_classification:
            pred = np.array(tf.math.argmax(pred))
        else:
            pred = pred[0]

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
        
        frame = cv.putText(frame, display_text, faces[0][:2], cv.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)


    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

print("Avg age: ", sum // count)