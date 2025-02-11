import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# This file shows how OpenCV and matplotlib load/read images

def main():
    # This statement loads the image with pillow and prints info.
    # OpenCV/matplot can't open an image loaded with Pillow.
    if True:
        raw1 = Image.open('small_flag.png')
        px = raw1.load()
        print("\nFirst Pixel Loaded by Pillow:", px[0,0])
        print("Pillow stores the true value for pixels.\n")
    
    # This statement loads the image with openCV.
    # It then shows how the loaded image looks 
    # when opened by OpenCV (normal) and matplotLib (red/blue inverted)
    if True:
        raw2 = cv.imread('small_flag.png')
        print("First Pixel Loaded by OpenCV:",raw2[0][0])
        print("OpenCV stores pixels by flipping their r and b values.\n")
        cv.imshow("Loaded with OpenCV, Opened by OpenCV",raw2)
        figA = plt.figure("Loaded with OpenCV, Opened by matplotlib")
        plt.imshow(raw2)
    
    # This statement loads the image with matplotlib.
    # It then shows how the loaded image looks 
    # when opened by OpenCV (red/blue inverted) and matplotLib (normal)
    if True:
        raw3 = mpimg.imread('small_flag.png')
        print("First Pixel Loaded by Matplt:", raw3[0][0])
        print("The rgb value here is the same as in Pillow, only scaled to between 0 and 1.\n")
        cv.imshow("Loaded with matplotlib, Opened by OpenCV",raw3)
        figB = plt.figure("Loaded with matplotlib, Opened by matplotlib")
        plt.imshow(raw3)

    # Open the windows
    plt.show()
    cv.waitKey(0)
    # Close the windows
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()