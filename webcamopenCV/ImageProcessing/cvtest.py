import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import argparse

def imageDrawing():
   # Create a black image
    img = np.zeros((512,512,3), np.uint8)
    #DrawLine args: Start,end,color,thickness
    cv.line(img,(0,0),(511,511),(255,0,0),5)
    #rectangle: bot left, top right, color, thickness
    cv.rectangle(img,(384,0),(510,128),(0,255,0),3)
    #Circle: center & radius
    cv.circle(img,(447,63), 63, (0,0,255), -1)
    cv.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
    pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)  #Drawing a polygon requires a list of vertices
    pts = pts.reshape((-1,1,2))
    cv.polylines(img,[pts],True,(0,255,255))
    # Write text onto a picture
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv.LINE_AA)
    
    cv.imshow('MyWindow',img)
    cv.waitKey(0) == ord('q')
    cv.destroyAllWindows()

def videoCapture():
    cap = cv.VideoCapture(0)    # Use 0 to capture video, replace it with a filename to play a file
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Display the resulting frame
        cv.imshow('frame', gray)
        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

def showImage():
    # Load an color image in grayscale
    img = cv.imread('messi5.jpg',0)
    #  1: cv.IMREAD_COLOR
    #  0: cv.IMREAD_GRAYSCALE
    # -1: cv.IMREAD_UNCHANGED

    cv.namedWindow('MyWindow', cv.WINDOW_NORMAL)   #cv.WINDOW_NORMAL lets you resize the window, defualt is WINDOW_AUTOSIZE
    cv.imshow('MyWindow',img)
    cv.waitKey(2000)  # Waits specifed miliseconds (indefinitely if 0) for any keypress before continuing
    cv.destroyWindow('MyWindow') #cv.destroyAllWindows
    print("Delete")
    return


def main():
    # showImage()
    # videoCapture()
    # imageDrawing()


if __name__ == "__main__":
    main()
