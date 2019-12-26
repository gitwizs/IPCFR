import numpy as np
import cv2
from mtcnn import MTCNN

detector = MTCNN()
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture('rtsp://admin:admin@172.20.48.19:8554/CH001.sdp')
#cap.set(3,640) # set Width
#cap.set(4,480) # set Height

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 0)
    face_locations = detector.detect_faces(img)
   

   for face in zip(face_locations):
        (x, y, w, h) = face[0]['box']
        landmarks = face[0]['keypoints']
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) 

    cv2.imshow('Capturing',img)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
