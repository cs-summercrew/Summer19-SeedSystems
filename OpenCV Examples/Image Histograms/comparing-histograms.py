# Authors: CS-World Domination Summer19 - DM
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import os.path
import shutil
import csv
from prettytable import PrettyTable
# NOTE: prettytable is only used to print the data so that it is more readable. It is otherwise unnecessary.
#       If you don't want to pip install prettytable then just comment out the respecitve code, it should all be in writeCSV()

def histCompare(baseFile, folderName, path):
    os.chdir(path)
    AllFiles = list(os.walk(path))[0][2]
    data = []
    # Take out the files we don't want to parse
    if '.DS_Store' in AllFiles:
        # NOTE: DS_Store files are an annoying Mac feature, if you aren't using MacOS you can delete this if statement
        ind = AllFiles.index('.DS_Store')
        AllFiles = AllFiles[:ind] + AllFiles[ind+1:]
    if baseFile in AllFiles:
        ind = AllFiles.index(baseFile)
        AllFiles = AllFiles[:ind] + AllFiles[ind+1:]
    # Create the histogram for our base image
    imageBase = cv.imread(baseFile, 1)  # -1 bgra, 0 gray, 1 bgr
    histBase = cv.calcHist([imageBase],[0,1,2],None,[256,256,256],[0, 256, 0, 256, 0, 256])
    cv.normalize(histBase, histBase, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)         # BUG CHECK!!!
    # NOTE: You could also use hsv to compare images:
    # imageBase = cv.cvtColor(imageBase, cv.COLOR_BGR2HSV)  
    for file in AllFiles:
        # Create the histograms for our comparison images
        imageToComp = cv.imread(file, 1) # -1 bgra, 0 gray, 1 bgr
        histToComp = cv.calcHist([imageToComp],[0,1,2],None,[256,256,256],[0, 256, 0, 256, 0, 256])
        cv.normalize(histToComp, histToComp, alpha=0, beta=1, norm_type=cv.NORM_MINMAX) # BUG CHECK!!!
        # Calculate comparisons and store to data
        result0 = cv.compareHist(histBase,histToComp,cv.HISTCMP_CORREL)
        result1 = cv.compareHist(histBase,histToComp,cv.HISTCMP_CHISQR)
        result2 = cv.compareHist(histBase,histToComp,cv.HISTCMP_INTERSECT)
        result3 = cv.compareHist(histBase,histToComp,cv.HISTCMP_BHATTACHARYYA)
        result4 = cv.compareHist(histBase,histToComp,cv.HISTCMP_KL_DIV)
        data.append([file, str(round(result0,4)), str(round(result1,4)), str(round(result2,4)), str(round(result3,4)), str(round(result4,4))])
    return data

def writeCSV(path, data):
    " Takes our data and writes it to a csv file"
    path = os.path.join(path, "HistogramData.csv")
    with open(path, 'w') as myCSVfile:
        print("Writing HistogramData.csv")
        filewriter = csv.writer(myCSVfile, delimiter='\n', quoting=csv.QUOTE_NONE, escapechar='\\')
        filewriter.writerow(["File_Name,Correlation,Chi-square,Intersection,BHATTACHARYYA,Kullback-Leibler"])
        t = PrettyTable(["File_Name", "Correlation", "Chi-square", "Intersection", "BHATTACHARYYA", "Kullback-Leibler"])
        for i in range(0,len(data)):
            filewriter.writerow([data[i][0] + ',' + data[i][1] + ',' + data[i][2] + ',' + data[i][3] + ',' + data[i][4] + ',' + data[i][5]])
            t.add_row([data[i][0], data[i][1], data[i][2], data[i][3][0], data[i][4], data[i][5]])
        print(t)
    return

def main():
    original_dir = os.getcwd()
    folderName = "960x640 Flowers"
    compFile = "f1.jpg"
    path = os.path.join(original_dir, folderName)
    data = histCompare(compFile, folderName, path)
    os.chdir(original_dir)
    writeCSV(original_dir, data)

if __name__ == "__main__":
    main()
