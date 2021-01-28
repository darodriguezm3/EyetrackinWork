import numpy as np
import pyautogui
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
from imutils import face_utils
import dlib
import time
import tensorflow as tf


#start=0
#vtiem=[]
# Vamos inicializar um detector de faces (HOG) para então
# let's go code an faces detector(HOG) and after detect the 
# landmarks on this detected face

# p = our pre-treined model directory, on my case, it's on the same script's diretory.
p = "AuxFiles/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


model1 = tf.keras.models.load_model('AuxFiles/saved_modelbywd1/my_model')
model2 = tf.keras.models.load_model('AuxFiles/saved_modelo4aux/my_model')
cap = cv2.VideoCapture(0)


while(True):
# Capture the video frame 
    # by frame 
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    # Get faces into webcam's image
    rects = detector(gray, 0)
    
    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
    	# Make the prediction and transfom it to numpy array
	    shape = predictor(gray, rect)
	    shape = face_utils.shape_to_np(shape)
	    o1=shape[36:41,:]
	    o2=shape[42:47,:]

    
	    # Draw on our image, all the finded cordinate points (x,y) 
	    '''
	    for (x, y) in shape:
	    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
	    '''
	    o1x=[int(max(0,np.min(o1[:,0])-15)),int(min(1280,np.max(o1[:,0])+15))]
	    o1y=[int(max(0,np.min(o1[:,1])-15)),int(min(1280,np.max(o1[:,1])+15))]

	    o2x=[int(max(0,np.min(o2[:,0])-15)),int(min(1280,np.max(o2[:,0])+15))]
	    o2y=[int(max(0,np.min(o2[:,1])-15)),int(min(1280,np.max(o2[:,1])+15))]


	    recortada1=gray[int(o1y[0]):int(o1y[1]), int(o1x[0]):int(o1x[1])]
	    recortada2=gray[int(o2y[0]):int(o2y[1]), int(o2x[0]):int(o2x[1])]

	    cv2.rectangle(frame,(o1x[0],o1y[0]),(o1x[1],o1y[1]),(255,0,0),2)
	    cv2.rectangle(frame,(o2x[0],o2y[0]),(o2x[1],o2y[1]),(255,0,0),2)

	    dim=(42,50)
	    normalizada1=cv2.resize(recortada1,(dim))/255
	    normalizada2=cv2.resize(recortada2,(dim))/255

	    normalizada1=np.reshape(normalizada1,(1,42,50,1))
	    normalizada2=np.reshape(normalizada2,(1,42,50,1))

	    c2=model1.predict(normalizada1)
	    c1=model2.predict(normalizada2)

	    cv2.circle(frame,(c1[0][0],c1[0][1]),10,(255,0,255),3)
	    cv2.circle(frame,(c2[0][0],c2[0][1]),10,(0,0,255),3)

	    pred=(c1+c2)/2
	    #cv2.circle(frame,(pred[0][0],pred[0][1]),10,(255,0,255),3)

	    print(pred)



	# Display the resulting frame 
    cv2.namedWindow('Grabando', cv2.WINDOW_NORMAL)
    cv2.imshow('Grabando', frame)   
    #stop=time.time()
    #act=stop-start
    #start=stop
    #vtiem.append(act)    
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        #np.savetxt('dlimod.csv',vtiem,delimiter=',')
        break

cv2.destroyAllWindows()
cap.release()