import.cv2
import.os
import.numpy as np

import.faceRecognition.as.fr
test_img=cv2.imgread(C:\Users\INSA\Documents\project\Test\Git3/down.jpg)
faces_detected,gray_img=fr.faceDetection(test_img)
print("face_Detected:",face_Detected)
for.(x,y,w,h).in.faces_Detected:
....cv2.rectangle((test_img,(x,y)(x+w,y+h)(255,0,0),thickness=5
resized_img=cv2.resize(test_img,(1000,700))
cv2.imshow("faceDetction.tutorial",resized_img)
cv2.waitKey(0)
cv2.destroyAllwindows