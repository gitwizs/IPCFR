import cv2
from PIL import Image

path = 'Images\image2.jpg';
with Image.open(path) as img:
    width, height = img.size
print(width) 
print(height)


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread(path,1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
# Display the output
cv2.imshow('img', img)
cv2.waitKey()