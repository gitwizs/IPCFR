''''
Real Time Face Recogition
	==> Each face stored on dataset/ dir, should have a unique numeric integer ID as 1, 2, 3, etc                       
	==> LBPH computed model (trained faces) should be on trainer/ dir
Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18  

'''

import cv2
import numpy as np
import os
from mtcnn import MTCNN

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')


detector = MTCNN()

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Alex', 'Aman', 'Mame', 'Negus', 'Fantsh' ,'Kalsh' ,'Membe' ,'Muler'] 

# Initialize and start realtime video capture
cam = cv2.VideoCapture('Images/10.mp4')
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height


while True:

    ret, img =cam.read()
    img = cv2.flip(img, 1) # Flip vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face_locations = detector.detect_faces(img)
    for face in zip(face_locations) :
         (x, y, w, h) = face[0]['box']
         landmarks = face[0]['keypoints']
         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
         id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
         # Check if confidence is less them 100 ==> "0" is perfect match 
         if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
         else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
         cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
         cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
 
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
