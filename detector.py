from http.client import ImproperConnectionState
import profile
import cv2
import numpy as np
import os
import pickle
import sqlite3
from PIL import Image
import streamlit as st


faceDetect = cv2.CascadeClassifier('haar_cascade.xml')
# cam = cv2.VideoCapture(0)
rec  = cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer/trainningData.yml")


def getProfile(id):
    conn= sqlite3.connect("database.db")
    cmd="SELECT * FROM people WHERE Id = " + str(id)
    cursor = conn.execute(cmd)
    profile = None
    for row in cursor:
        profile=row
        conn.close()
        return profile

id = 0
font = cv2.FONT_HERSHEY_SIMPLEX
def detect_faces(our_image):
    img = np.array(our_image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = faceDetect.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    name='Unknown'
    for (x, y, w, h) in faces:
        # To draw a rectangle in a face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        id, uncertainty = rec.predict(gray[y:y + h, x:x + w])
        print(id, uncertainty)
        profile= getProfile(id)

        if(profile!= None):
            #cv2.putText(img,str(profile[0]) ,(x,y+h+30), font,0.75,(0,255,0),2)
            cv2.putText(img,"Name : "+str(profile[1]) ,(x,y+h+60), font,0.75,(0,255,0),2)
            cv2.putText(img,"AGE : "+str(profile[2]) ,(x,y+h+90), font,0.75,(0,255,0),2)
            cv2.putText(img,"Status :"+str(profile[3]) ,(x,y+h+120), font,0.75,(0,255,0),2)
        else:
            cv2.putText(img,"Unknown" ,(x,y+h+60), font,0.75,(0,255,0),2)



    return img



# while(True):
#     ret,img = cam.read()
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     faces = faceDetect.detectMultiScale(gray , 1.3 , 5)
#     for(x,y,w,h) in faces:
#         cv2.rectangle(img,(x,y),(x+w,y+h) , (0,255,0),2)
#         id,conf =rec.predict(gray[y:y+h,x:x+w])
        
#         profile= getProfile(id)
#         if(profile!= None):
#             #cv2.putText(img,str(profile[0]) ,(x,y+h+30), font,0.75,(0,255,0),2)
#             cv2.putText(img,"Name : "+str(profile[1]) ,(x,y+h+60), font,0.75,(0,255,0),2)
#             cv2.putText(img,"AGE : "+str(profile[2]) ,(x,y+h+90), font,0.75,(0,255,0),2)
#             cv2.putText(img,"Status :"+str(profile[3]) ,(x,y+h+120), font,0.75,(0,255,0),2)

#     cv2.imshow("Face",img)
#     if(cv2.waitKey(1) == ord('q')):
#         break

# cam.release()
# cv2.destroyAllWindows()

def main():
    """Face Recognition App"""

    st.write("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

        
        body,[class*="css"]  {
            font-family: "Poppins", sans-serif;
            background-color: #1f1f38;

        }

        </style>
    """, unsafe_allow_html=True)
    

    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)

    st.title("")

    html_temp = """
    <body>
    <div className="heading" >
    <h2 style="color:#4db5ff;text-align:center;">Detect Vaccination Status</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if image_file is not None:
        our_image = Image.open(image_file)
        st.text("Original Image")
        st.image(our_image)

    if st.button("Recognise"):
        result_img= detect_faces(our_image)
        st.image(result_img)


if __name__ == '__main__':
    main()