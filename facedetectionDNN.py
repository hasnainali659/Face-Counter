# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 12:23:52 2020

@author: Hasnain
"""

import cv2
import dlib
from imutils import face_utils
def detectVidDlibCNN(vidpath):

    video_capture = cv2.VideoCapture(vidpath)
    flag = 0
    dnnFaceDetector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('streetDNN.avi',fourcc,20.0,(int(video_capture.get(3)),int(video_capture.get(4))))
    
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        #frame  = cv2.resize(frame,(512,512))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = dnnFaceDetector(gray, 1)
        
        j = '1'
        for (i, rect) in enumerate(rects):

            x1 = rect.rect.left()
            y1 = rect.rect.top()
            x2 = rect.rect.right()
            y2 = rect.rect.bottom()

            # Rectangle around the face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame,j,(x1+2,y1+150), cv2.FONT_HERSHEY_SIMPLEX , 4, (0,255,0),15, cv2.LINE_AA)
            j = int(j)+1
            j = str(j)

        # Display the video output
        out.write(frame)
        #cv2.imshow('Video', frame)

        # Quit video by typing Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    video_capture.release()
    cv2.destroyAllWindows()

   
    
detectVidDlibCNN('street.mp4')