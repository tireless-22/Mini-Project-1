# Mainly used things
# cascade c


import cv2
# opencv
import os
from keras.models import load_model

import numpy as np
from pygame import mixer
# mixer pygame module for loading and playing sounds is available 
# and initialized before using it. 
# The mixer module has a limited number of channels for playback of sounds. 
# Usually programs tell pygame to start playing audio and it selects an available channel automatically.
import time



mixer.init()
sound = mixer.Sound('alarm.wav')

# A Haar classifier, or a Haar cascade classifier, 
# is a machine learning object detection program that 
# identifies objects in an image and video
face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')



lbl=['Close','Open']

model = load_model('models/cnncat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

while(True):
    ret, frame = cap.read()
    # ret is a boolean variable that returns true if the frame is available.
    # frame is an image array vector captured based on the default frames per second defined explicitly or implicitly
    height, width, channels = frame.shape
    # frame.shape returns 3 values height ,width and channels we need only the height and width

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # we have to convert the color image into dark image
    # Grayscale images are monochrome images, Means they have only one color. 
    # 0000000Grayscale images do not contain any information about color. ... A normal grayscale image contains 8 bits/pixel data
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    # it will return all face values in that frame
    # for every face this will return 4 values
    # x,y,w,h
    # x indicates where the object started in the x-axis
    # y indicates where the object started in the y-axis
    # w indicates the width of that object
    # h indicates the height of that object
    # objects = cv.CascadeClassifier.detectMultiScale(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]]
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    

    for (x,y,w,h) in faces:

        if(len(left_eye)):

            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,256,0) , 3 )
        else:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,165,255) , 3 )

        #  cv2.rectangle(image, start_point, end_point, color, 2)

    

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,256,0) ,1 )

        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = np.argmax(model.predict(r_eye), axis=-1)

       
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,256,0) , 1 )
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = np.argmax(model.predict(l_eye), axis=-1)
      
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break

    if(rpred[0]==0 and lpred[0]==0 and len(faces)!=0):
        score=score+1
        # inorder to display the closed  score 
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(0,0,0),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):

    else:
        score=score-1
       # inorder to display the closed  ror open 
        cv2.putText(frame,"Open",(10,height-20), font, 1,(0,0,0),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(0,0,0),1,cv2.LINE_AA)
    if(score>8):
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()
            
        except:  # isplaying = False
            pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
        cv2.putText(frame,"Drowsy",(10,20), font, 1,(0,0,0),1,cv2.LINE_AA)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # this will break the loop if we enter the q letter
        break
    # 
cap.release()
# it is to release the camera

cv2.destroyAllWindows()
# it is to destroy the windows


