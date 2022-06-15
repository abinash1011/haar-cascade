import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier('project1\haarcascades\haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('project1\haarcascades\haarcascade_eye.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = face_classifier.detectMultiScale(gray,1.3,5)


    for (x, y , w, h) in detections:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        face_gray = gray[y:y+h, x:x+h]
        face_color = frame[y:y+h, x:x+h]
        eyes = eye_classifier.detectMultiScale(face_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_color, (ex, ey), (ex+ew, ey+eh), (0,0,255), 2)
        
        

    # if (len(detections)>0):
    #     (x,y,w,h) = detections[0]
    #     frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        break
