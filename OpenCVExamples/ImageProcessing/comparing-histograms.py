import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import os.path
import shutil

def histCompare():
    #https://docs.opencv.org/2.4.13.6/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html?highlight=histogram
    # Do something showing off these functions
    cv.compareHist(x,x,CV_COMP_CORREL)
    # cv2.HISTCMP_CHISQR
    # cv2.HISTCMP_INTERSECT
    # cv2.HISTCMP_BHATTACHARYYA
    # TODO: use the different metrics on a folder of images to compare to a single image
    return

def main():
    #histCompare()
    pass

if __name__ == "__main__":
    main()
