import sqlite3
import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('haar_cascade.xml')
cam = cv2.VideoCapture(0)

def insertOrUpdate(Id,Name,Age,Vs , mobNo , add):
    conn=sqlite3.connect("database.db")
    cmd="SELECT * FROM people WHERE Id="+str(Id)
    cursor=conn.execute(cmd)
    isRecordExist = 0
    for row in cursor:
        isRecordExist = 0
    if(isRecordExist ==1):
        cmd = "UPDATE people SET Name=' " + str(Name) + " ' WHERE Id ="+str(Id)
    else:
        cmd="INSERT INTO people VALUES ("+str(Id) +" , ' " + str(Name)  + " ' ,  " + str(Age)  + "  , ' " + str(Vs)  + " ' , ' " + str(mobNo)  + " ' , ' " + str(add)  + " ' )"
    conn.execute(cmd)
    conn.commit()
    conn.close()


id = input('Enter User id ')
name = input('Enter Your Name ')
age = input("Enter Your Age ")
vs = input("Enter Your Vaccination Status : \n Fully Vaccinated \n Partially Vaccinated \n Not Vaccinated : \n ")
mobNo = input("Enter Your Mobile Number : ")
add = input("Enter Your Address : ")


insertOrUpdate(id,name,age,vs , mobNo,add)

sampleNum = 0
while(True):
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray , 1.3 , 5)
    for(x,y,w,h) in faces:
        sampleNum = sampleNum +1
        cv2.imwrite("dataSet/User."+str(id) + "." + str(sampleNum) +".jpg",gray[y:y+h , x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h) , (0,0,255),2)
        cv2.waitKey(100)
    cv2.imshow("Face",img)
    if(cv2.waitKey(1)):
        if sampleNum >20:
            break

cam.release()
cv2.destroyAllWindows()