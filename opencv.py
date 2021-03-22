#Testing the model in Real-time using OpenCV and WebCam
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json


model = model_from_json(open("model.json", "r").read())
model.load_weights('model.h5')

eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)

emotionsdict = {0:'anger',1:'fear',2:'happy',3:'sadness',4:'surprise'}

while True:
    ret , img = cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]
        eyes=eye_cascade.detectMultiScale(roi_gray,1.3,5)
        roi_gray = np.resize(roi_gray,(1,48,48,1))


        image_pixels = np.array(roi_gray)



        predictions = model.predict(image_pixels)
        idx = predictions.argmax()

        emotion_detection = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        emotion_prediction = emotion_detection[idx]

        cv2.putText(img, emotion_prediction ,(x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 255), 2, cv2.LINE_AA)
        for (ex,ey,ew,eh ) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow('img',img)
    k= cv2.waitKey(30)& 0xff
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()

