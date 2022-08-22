import cv2
import sys
import os

os.chdir(r'/Users/madelinelui/Desktop/python/face-recognition')

imagePath = sys.argv[1]
cascPath = sys.argv[2]

#haar cascade
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

image = cv2.imread(imagePath)
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#detect
faces = faceCascade.detectMultiScale(
    grey,
    scaleFactor=1.115,
    minNeighbors=5,
    minSize=(30,30),
    flags=cv2.CASCADE_SCALE_IMAGE
)

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x,y), (x+w, y+h),(0,255,0),2)

print("Found {0} faces!".format(len(faces)))

cv2.imshow("Faces found", image)
cv2.waitKey(0)

#python face_recog.py andre-test.png haarcascade_frontalface_default.xml