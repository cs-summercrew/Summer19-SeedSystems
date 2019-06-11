import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import os.path
import shutil

# NOTE: The helper function hist_match is taken directly from the following stackoverflow post. 
#       Apparently there are no libraries that do color-histogram matching.
#       https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
# 
#       The link below is also probably worth taking a look at, if you want to use MATLAB's matching function instead.
#       https://www.mathworks.com/help/matlab/matlab-engine-for-python.html?searchHighlight=python&s_tid=doc_srchtitle


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
            line2, = plt.plot(hist2,label="Equalized Data",color='brown')
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

def histCompare():
    #https://docs.opencv.org/2.4.13.6/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html?highlight=histogram
    # Do something showing off these functions
    file1="pug.png"
    img1= cv.imread(file1, 0)
    file2="/Banana/frame600.png"
    img2= cv.imread(file2, 0)
    hist1 = cv.calcHist([img1],[0],None,[256],[0,256])
    hist2 = cv.calcHist([img2],[0],None,[256],[0,256])
    results = cv.compareHist(hist1,hist2,cv.HISTCMP_CORREL) # Show off the different metrics
    print (results)
    return

def histColorMatch(srcFile, tempFile):
    # NOTE: See https://en.wikipedia.org/wiki/Histogram_matching for a description of this algorithm
    imageSrc = cv.imread(srcFile, 0) # -1 alpha, 0 gray, 1 color
    imageTemp = cv.imread(tempFile, 0) # -1 alpha, 0 gray, 1 color
    imageMatch = hist_match(imageSrc, imageTemp).astype(np.uint8)
    while True:
        k = cv.waitKey(10) & 0xFF
        k_char = chr(k)
        if k_char == ' ':
            print("Displaying histogram!")
            hist1 = cv.calcHist([imageSrc],[0],None,[256],[0,256])
            hist2 = cv.calcHist([imageTemp],[0],None,[256],[0,256])
            hist3 = cv.calcHist([imageMatch],[0],None,[256],[0,256])
            line3, = plt.plot(hist3,label="MatchResult Data",color='blue')
            line2, = plt.plot(hist2,label="Template Data",color='red')
            line1, = plt.plot(hist1,label="Source Data",color='black')
            first_legend = plt.legend(handles=[line2], loc=1)
            ax = plt.gca().add_artist(first_legend)
            plt.legend(handles=[line1], loc=4)
            plt.xlim([0,256])
            plt.show()
        # Display the resulting image
        cv.imshow("srcFile", imageSrc)
        cv.imshow("tempFile", imageTemp)
        cv.imshow("matchResult", imageMatch)
        # End the Video Capture
        if k == 27: # ESC key, See https://keycode.info for other keycodes
            print("Closed image windows!")
            break
    cv.destroyAllWindows()
    return

def hist_match(source, template):
    # NOTE: This function was taken from the stake overflow post
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

def main():
    # histBGR("messi5.jpg")
    # histEqual("treelc.jpg")
    # histColorMatch("f1.jpg", "f5.jpg")
    # histCompare()
    # histColorMatch()

if __name__ == "__main__":
    main()
