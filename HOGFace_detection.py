import cv2
import dlib
from imutils import face_utils

def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)



def detectVidHOG(vidpath):
    video_capture = cv2.VideoCapture(vidpath)
    flag = 0

    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #out = cv2.VideoWriter('face.avi',fourcc,20.0,(int(video_capture.get(3)),int(video_capture.get(4))))
    while True:

        ret, frame = video_capture.read()
        gamma = 1.5                                   # change the value here to get different result
        frame = adjust_gamma(frame, gamma=gamma)
        #frame = cv2.resize(frame,(256,256))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detect = dlib.get_frontal_face_detector()

        rects = face_detect(gray, 1)
        
        j = '1'
        for (i, rect) in enumerate(rects):

            (x, y, w, h) = face_utils.rect_to_bb(rect)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(frame,j,(x+2,y+150), cv2.FONT_HERSHEY_SIMPLEX , 4, (0,255,0),15, cv2.LINE_AA)
            j = int(j)+1
            j = str(j)
        #out.write(frame)
        #cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    
detectVidHOG('C:/Users/DELL/OneDrive for Business/Desktop/cv_assigment/CV_project/sample_videos/classroom.mp4')