#"Testing Images"

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

#Load Classifier
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load an color image in grayscale
img = cv2.imread('Images\image2.jpg',1)
scale_percent = 30 # percent of original size
# width = int(700)
# height = int(700)
# dim = (width, height)
# resize image
# img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

#img = cv2.flip(img, -1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
# important parameters on detected face.
faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.1,
        minNeighbors=5,     
        minSize=(20, 20)
    )
#Draw Rectangle in Detected faces
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    
cv2.imshow('image',img)

#cv2.imshow('image',img)
k = cv2.waitKey(0)
if(k == 27) :
    cv2.destroyAllWindows()
elif ( k == ord('s')) :
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    #cv2.destroyAllWindows()