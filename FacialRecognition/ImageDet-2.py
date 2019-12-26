import cv2
from PIL import Image
from mtcnn import MTCNN
#import face_recognition

path = 'Images\image2.jpg';
with Image.open(path) as img:
    width, height = img.size
print(width) 
print(height)

detector = MTCNN()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread(path,1)
face_locations = detector.detect_faces(img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
# Display the output
cv2.imshow('Haarcascade ', img)

for face in zip(face_locations):
    (x, y, w, h) = face[0]['box']
    landmarks = face[0]['keypoints']
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    # for key, point in landmarks.items():
        # cv2.circle(img, point, 2, (255, 0, 0), 6)

cv2.imshow('Multi-task Cascaded Convolutional Networks',img)
cv2.waitKey()