import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import argparse

def videoCapture():
    cap = cv.VideoCapture(0)    
    #NOTE: Use as an arg 0 to capture video with the default camera, using a filename as an arg plays the file
    #      If you have more than one camera device, selects the second by passing 1, 2, and so on...
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    currFrame = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        print(currFrame)
        currFrame +=1
        # Our SAVED operations on the frame go here
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Write the current frame to a file
        if cv.waitKey(50) == ord('s'):
            cv.imwrite("frame"+str(currFrame)+".png", frame)
        # Our UNSAVED operations on the frame go here
            #TODO: This is where we write text
        
        # Display the resulting frame
        cv.imshow('frame', frame)
        # End the Video Capture
        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


def main():
    videoCapture()

if __name__ == "__main__":
    main()
