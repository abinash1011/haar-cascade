import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier('project1\haarcascades\haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, 0)
    detections = face_classifier.detectMultiScale(gray,1.3,5)

    if (len(detections)>0):
        (x,y,w,h) = detections[0]
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        break
