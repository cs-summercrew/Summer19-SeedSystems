import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import os.path
import shutil


def videoCapture(path):
    cap = cv.VideoCapture(0)    
    #NOTE: Using 0 as an arg captures video with the default camera, using a filename as an arg plays the file
    #      If you have more than one camera device, use the these other cameras by passing 1, 2, and so on...
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    currFrame = 0
    color = True
    contSave = False
    limit = 0
    os.chdir(path)  # Changes the cwd to the path we print files to
    while True:
        ret, frame = cap.read()
        frame = cv.resize(frame, None, fx=.5, fy=.5)   # Resizes the frame (smaller runs faster)
        # NOTE: You can edit each frame of the video with all the same OpenCV methods as an image
        if not ret:
            # if frame is read correctly ret is True
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # print(currFrame)
        currFrame +=1
        # Our SAVED operations on the frame go here
        k = cv.waitKey(20) & 0xFF
        # NOTE: "& 0xFF" is used because NumLock does weird things to keycodes according to StackOverflow
        # NOTE: If you use more than one cv.waitKey(), your program will slow down significantly...
        k_char = chr(k)
        if k_char == 's':
            color = not color
            print("You switched the color display!")
        if not color:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if (k_char == ' ') and (not contSave):
            cv.imwrite("frame"+str(currFrame)+".png", frame)
            print("You saved frame "+str(currFrame)+"!")
        if (k_char == 'c'):
            contSave = not contSave
            print("You saved frames "+str(currFrame)+"-"+str(currFrame+30)+"!")
        if contSave:
            cv.imwrite("frame"+str(currFrame)+".png", frame)
            limit+=1
            if(limit > 30):
                contSave = not contSave
                limit = 0

        # Our UNSAVED operations on the frame go here
        cv.rectangle(frame,(0,0),(300,105),(0,0,0),-1)   #Setting the last arg (pixel width) fills the rectangle
        cv.putText(frame,'Frame: '+str(currFrame),(10,45),cv.FONT_HERSHEY_PLAIN,2.5,(255,255,255),2,cv.LINE_AA)
        if color:
            temp = "Color"
        else:
            temp = "Gray"
        cv.putText(frame,'Display: '+temp,(10,90),cv.FONT_HERSHEY_PLAIN,2.5,(255,255,255),2,cv.LINE_AA)
        # NOTE: See avaliable fonts at: https://www.docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga0f9314ea6e35f99bb23f29567fc16e11
        
        # Display the resulting frame
        cv.imshow('frame', frame)
        # End the Video Capture
        if k == 27: # ESC key, See https://keycode.info for other keycodes
            print("End of Video Capture")
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    return

def setup(original_dir):
    " Makes a folder to hold our frame files, and return a path to that folder"
    path = os.path.join(original_dir, "temp")
    try:
        os.mkdir(path)
    except:
        print("Didn't make the folder because it already exists!")
    return path

def main():
    original_dir = os.getcwd()
    path = setup(original_dir)
    videoCapture(path)
    os.chdir(original_dir)  # Changes cwd back to original_dir

if __name__ == "__main__":
    main()
