import cv2 as cv
import numpy as np

faceDetect = cv.CascadeClassifier('haar_cascade.xml')
cam = cv.VideoCapture(0)

while(True):
    ret,img = cam.read()
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray , 1.3 , 5)
    for(x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h) , (0,0,255),2)
    cv.imshow("Face",img)
    if(cv.waitKey(1) == ord('q')):
        break

cam.release()
cv.destroyAllWindows()