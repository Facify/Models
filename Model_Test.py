import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

cap = cv.VideoCapture(0)
model_path = r'./checkpoints/age_range_model1.epoch16-loss4.51.hdf5'
model = tf.keras.models.load_model(model_path)

is_age_range = "age_range" in model_path

count = 0
sum = 0

while True: 
    ret, frame = cap.read()
    sq_img = np.array([row[:480] for row in np.asarray(frame)])
    gray = cv.cvtColor(sq_img, cv.COLOR_BGR2GRAY)
    small = cv.resize(gray, (48, 48))
    cv.imshow('frame', sq_img)

    img = np.asarray(small).reshape((48,48,1))
    plt.imshow(img.reshape(48,48))
    pred = model.predict(np.array([img]))[0][0] 

    if pred != 0:
        count += 1
        if is_age_range:
            sum += 5 * pred
            print("predicted age range is:", pred * 5, "to", (pred + 1) * 5)
        else: 
            sum += pred
            print("Predicted age: ", pred)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

print("Avg age: ", sum // count)