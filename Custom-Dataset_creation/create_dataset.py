from imutils.video import VideoStream
import argparse 
import cv2 as cv
import numpy as np 
import os
import time
import imutils
#parsing arguments

ap = argparse.ArgumentParser()

ap.add_argument("-c","--cascade",required=True,
help="path to cascade classifier")
ap.add_argument("-o","--output",required=True,
help="path to where the screenshots will be stored.")

args = vars(ap.parse_args())


detector  = cv.CascadeClassifier(args["cascade"])
#for detection from a video we can give add path to a video instead 
#of using 
print("INFO starting videostream")
vs = VideoStream(src=0).start()

time.sleep(2.0)
total = 0

while True:

    frame = vs.read()
    orig = frame.copy()
    frame = imutils.resize(frame,width=400)
    rects = detector.detectMultiScale(cv.cvtColor(frame,cv.COLOR_RGB2GRAY),scaleFactor=1.1,minNeighbors=5,minSize=(30,30))

    for (x,y,w,h) in rects:
        cv.rectangle(frame,(x,y),(x+w,y+h),(14,244,0),2)
        cv.imshow("Frame",frame)
        key = cv.waitKey(1) & 0xFF

        if key == ord("k"):#press k to store image in the output directory(take screenshots.)
            p = os.path.sep.join([args["output"],"{}.png".format(str(total).zfill(5))])
            cv.imwrite(p,orig)
            total+=1
        elif key == ord("q"):
            break


print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up .. .. ")

cv.destroyAllWindows()
vs.stop()