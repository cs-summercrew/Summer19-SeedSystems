import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import os.path
import shutil


def histBGR(file):
    "Press space for color-intensity histogram of the image's bgr values"
    image = cv.imread(file, 1) # -1 alpha, 0 gray, 1 color
    # hist = cv.calcHist([image],[0],None,[256],[0,255])
    # plt.hist(image.ravel(),256,[0,256])
    while True:
        k = cv.waitKey(10) & 0xFF
        k_char = chr(k)
        if k_char == ' ':
            print("Displaying histogram!")
            color = ['b','g','r']
            for i,col in enumerate(color):
                histr = cv.calcHist([image],[i],None,[256],[0,256])
                # NOTE: Using an argument other than None lets you creates a hist for a specified section of the image
                plt.plot(histr,color = col)
                plt.xlim([0,256])
            plt.show()
        # Display the resulting image
        cv.imshow(file, image)
        # End the Video Capture
        if k == 27: # ESC key, See https://keycode.info for other keycodes
            print("Closed image windows!")
            break
    cv.destroyWindow(file)
    return

def histEqual(file):
    "Returns the image result of equalizehist: makes the Equalized Data more spread out than Original Data"
    image = cv.imread(file, 0) # -1 alpha, 0 gray, 1 color
    EQimage = cv.equalizeHist(image)    # Equalization only works with grayscale images
    while True:
        k = cv.waitKey(10) & 0xFF
        k_char = chr(k)
        if k_char == ' ':
            print("Displaying histogram!")
            hist1 = cv.calcHist([image],[0],None,[256],[0,256])
            hist2 = cv.calcHist([EQimage],[0],None,[256],[0,256])
            line2, = plt.plot(hist2,label="Equalized Data",color='red')
            line1, = plt.plot(hist1,label="Original Data",color='black',linestyle='dashed')
            first_legend = plt.legend(handles=[line2], loc=1)
            ax = plt.gca().add_artist(first_legend)
            plt.legend(handles=[line1], loc=4)
            plt.xlim([0,256])
            plt.show()
        # Display the resulting image
        cv.imshow(file, image)
        cv.imshow("EQ_"+file, EQimage)
        # Close open image windows
        if k == 27: # ESC key, See https://keycode.info for other keycodes
            print("Closed image windows!")
            break
    cv.destroyWindow(file)
    cv.destroyWindow("EQ_"+file)
    return EQimage

def histColorMatch():
    #https://docs.opencv.org/2.4.13.6/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html?highlight=histogram
    # Do something showing off these functions
    cv.compareHist(x,x,CV_COMP_CORREL) # Show off the different metrics

def main():
    # histBGR("messi5.jpg")
    # histEqual("treelc.jpg")
    # histCompare()
    # histColorMatch()

if __name__ == "__main__":
    main()
