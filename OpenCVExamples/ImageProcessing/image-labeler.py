import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import os.path
import shutil


def labeler(path):
    os.chdir(path)  # Changes the cwd
    FILE_NAME = "frame448.png"
    currImage = cv.imread(FILE_NAME)
    while True:
        
        k = cv.waitKey(20) & 0xFF
        k_char = chr(k)
        if k_char == 's':
            color = not color
            print("You switched the color display!")
        # Our UNSAVED operations on the frame go here
        cv.rectangle(currImage,(0,0),(310,40),(0,0,0),-1)   #Setting the last arg (pixel width) fills the rectangle
        cv.putText(currImage,FILE_NAME,(5,30),cv.FONT_HERSHEY_PLAIN,2.5,(255,255,255),2,cv.LINE_AA)
        # NOTE: See avaliable fonts at: https://www.docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga0f9314ea6e35f99bb23f29567fc16e11
        
        # Display the resulting frame
        cv.imshow(FILE_NAME, currImage)
        # End the Video Capture
        if k == 27: # ESC key, See https://keycode.info for other keycodes
            print("Moving on to the next Image!")
            break

    cv.destroyAllWindows()
    return

def main():
    original_dir = os.getcwd()
    folderName = "Bananna"
    path = os.path.join(original_dir, folderName)
    labeler(path)
    os.chdir(original_dir)  # Changes cwd back to original_dir

if __name__ == "__main__":
    main()
