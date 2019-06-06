import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def videoCapture():
    cap = cv.VideoCapture(0)    
    #NOTE: Use as an arg 0 to capture video with the default camera, using a filename as an arg plays the file
    #      If you have more than one camera device, selects the second by passing 1, 2, and so on...
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    currFrame = 0
    currMode = 'Gray'
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
        if cv.waitKey(25) == 83: # S key
            # NOTE: Go to https://keycode.info and press any key on your keyboard to get its key code
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            print("You switched color modes!")
        # Save the current frame to a file
        if cv.waitKey(25) == 32: # SPACEBAR
            cv.imwrite("frame"+str(currFrame)+".png", frame)
            print("You saved frame "+str(currFrame)+"!")
        # Our UNSAVED operations on the frame go here
        cv.rectangle(frame,(0,0),(320,105),(0,0,0),-1)   #Setting the last arg (pixel width) fills the rectangle
        cv.putText(frame,'Frame:'+str(currFrame),(10,45),cv.FONT_HERSHEY_PLAIN,3,(255,255,255),2,cv.LINE_AA)
        cv.putText(frame,'Mode: '+str(currMode),(10,90),cv.FONT_HERSHEY_PLAIN,3,(255,255,255),2,cv.LINE_AA)
        # NOTE: See avaliable fonts at: https://www.docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga0f9314ea6e35f99bb23f29567fc16e11
        
        # Display the resulting frame
        cv.imshow('frame', frame)
        # End the Video Capture
        if cv.waitKey(1) == 27: # ESC key
            print("End of Video Capture")
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


def main():
    videoCapture()

if __name__ == "__main__":
    main()
