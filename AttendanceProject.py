import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path = 'ATTENDANCE IMAGE'
images=[]
classNames = []
myList = os.listdir(path)
print(myList)
for c1 in myList:
      curImg = cv2.imread(f'{path}/{c1}')
      images.append(curImg)
      classNames.append(os.path.splitext(c1)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList=[]
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')



encodeListKnown = findEncodings(images)
print('Encoding Complete')
# capturing the new images in the camera
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_RGB2BGR)

   # scale_percent = 60  # percent of original size
   # width = int(img.shape[1] * scale_percent / 100)
   # height = int(img.shape[0] * scale_percent / 100)
   # dim = (width, height)

    # resize image
  #  resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # frameRGB = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
  #  cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
    #checking if images are same or not
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
              name = classNames[matchIndex].upper()
              print(name)

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        markAttendance(name)

        cv2.imshow('Webcam', img)
        cv2.waitKey(1)

#y1,x1,y2,x2 = faceLoc
#y1, x1, y2, x2 =  y1*4,x1*4,y2*4,x2*4


#img = cv2.imread('E:/Images/ece/1.png',1)
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#faceLoc = face_recognition.face_locations(imgElon)[0]
#encodeElon = face_recognition.face_encodings(imgElon)[0]
#cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

#faceLocTest = face_recognition.face_locations(imgTest)[0]
#encodeTest = face_recognition.face_encodings(imgTest)[0]
#cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

#results = face_recognition.compare_faces([encodeElon],encodeTest)
#to find how similar two  images are use distance , lower the distance value more the similarity
#faceDis = face_recognition.face_distance([encodeElon],encodeTest)