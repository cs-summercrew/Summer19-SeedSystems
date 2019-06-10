import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import os.path
import shutil


def hist(file):
    "Takes the path to a folder of images, loops through displaying iamges, and ask for human input, which is outputted as a list"
    image = cv.imread(file)
    color = True
    while True:
        k = cv.waitKey(10) & 0xFF
        k_char = chr(k)
        if k_char == 's':
            color = not color
            print("You switched the color display!")
        if not color:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # Display the resulting image
        cv.imshow(file, image)
        # End the Video Capture
        if k == 27: # ESC key, See https://keycode.info for other keycodes
            print("Closed image window!")
            break
    cv.destroyWindow(file)
    return

def main():
    hist("RaybansPug.png")
    cv.destroyAllWindows() # Deletes any opened windows just in case

if __name__ == "__main__":
    main()
