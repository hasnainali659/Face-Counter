# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:43:44 2020

@author: DELL
"""

import cv2
import numpy as np


def detect(gray, frame): # We create a function that takes as input the image in black and white (gray) and the original image (frame), and that will return the same image with the detector rectangles. 
    faces = face_cascade.detectMultiScale(gray, 1.3, 8) # We apply the detectMultiScale method from the face cascade to locate one or several faces in the image.
    i = '1'
    for (x, y, w, h) in faces: # For each detected face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 10) # We paint a rectangle around the face.
        cv2.putText(frame,i,(x+2,y+150), cv2.FONT_HERSHEY_SIMPLEX , 4, (0,255,0),15, cv2.LINE_AA)
        i = int(i)+1
        i = str(i)
    return frame,faces

def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

cap = cv2.VideoCapture('C:/Users/DELL/OneDrive for Business/Desktop/cv_assigment/CV_project/sample_videos/classroom.mp4')
frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))

#fourcc = cv2.VideoWriter_fourcc('X','V','I','D')

#out = cv2.VideoWriter("output.mp4", fourcc, 5.0, (1280,720))
#out = cv2.VideoWriter(
#    'output.avi',
#    cv2.VideoWriter_fourcc(*'MJPG'),
#    15.,
#    (640,480))

#out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))

while cap.isOpened():
    ret, frame = cap.read()
    gamma = 1.5                                   # change the value here to get different result
    frame = adjust_gamma(frame, gamma=gamma)
    #frame = frame + 10
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('C:/Users/DELL/OneDrive for Business/Desktop/cv_assigment/CV_project/models/haarcascade_frontalface_default.xml')
    #full_body = cv2.CascadeClassifier('haarcascade_fullbody.xml')
    canvas,faces = detect(gray, frame)
    canvas_resize = image_resize(canvas, height = 600)
    cv2.imshow('image', canvas_resize)
    
    #out.write(frame)
    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
#out.release()