import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import os.path
import shutil
import csv


def labeler(path, label):
    "Takes the path to a folder of images, loops through displaying iamges, and ask for human input, which is outputted as a list"
    os.chdir(path)  # Changes the cwd
    labelList = []
    AllFiles = list(os.walk(path))[0][2]
    if '.DS_Store' in AllFiles:
        # NOTE: DS_Store files are an annoying Mac feature, if you aren't using MacOS you can delete this if statement
        ind = AllFiles.index('.DS_Store')
        AllFiles = AllFiles[:ind] + AllFiles[ind+1:]
    for file in AllFiles:
        has_attr = False
        hasNo_attr = False
        currImage = cv.imread(file)
        while True:
            cv.rectangle(currImage,(0,0),(310,40),(0,0,0),-1)   #Setting the last arg (pixel width) fills the rectangle
            if (not has_attr) and (not hasNo_attr):
                cv.putText(currImage,"Press y or n",(5,30),cv.FONT_HERSHEY_PLAIN,2.5,(255,255,255),2,cv.LINE_AA)
            if has_attr:
                cv.putText(currImage,"Yes: "+label,(5,30),cv.FONT_HERSHEY_PLAIN,2.5,(255,255,255),2,cv.LINE_AA)
            if hasNo_attr:
                cv.putText(currImage,"No: "+label,(5,30),cv.FONT_HERSHEY_PLAIN,2.5,(255,255,255),2,cv.LINE_AA)
            k = cv.waitKey(20) & 0xFF
            k_char = chr(k)
            if k_char == 'n':
                has_attr = False
                hasNo_attr = True
                print("You marked "+file+" as not having "+label)
            if k_char == 'y':
                has_attr = True
                hasNo_attr = False
                print("You marked "+file+" as having "+label)
            # Display the resulting frame
            cv.imshow(file, currImage)
            # End the Video Capture
            if k == 27: # ESC key, See https://keycode.info for other keycodes
                if has_attr:
                    labelList.append([file, "1"])
                    print("You saved the label for "+file+"!")
                if hasNo_attr:
                    labelList.append([file, "0"])
                    print("You saved the label for "+file+"!")
                if (not has_attr) and (not hasNo_attr):
                    labelList.append([file, "N/A"])
                    print("You did not choose a label for "+file+"!")
                print("Moving on to the next Image!")
                break
        cv.destroyWindow(file)
    return labelList

def writeCSV(path, data):
    " Takes our data and writes it to a csv file"
    path = os.path.join(path, "BanannaData.csv")
    with open(path, 'w') as myCSVfile:
        print("Writing BanannaData.csv")
        filewriter = csv.writer(myCSVfile, delimiter='\n', quoting=csv.QUOTE_NONE, escapechar='\\')
        filewriter.writerow(["File_Name,Has_Attribute"])
        for i in range(0,len(data)):
                filewriter.writerow([data[i][0] + ',' + data[i][1]])
    return

def main():
    original_dir = os.getcwd()
    folderName = "Bananna"
    path = os.path.join(original_dir, folderName)
    labelList = labeler(path, "Bannana")
    os.chdir(original_dir)  # Changes cwd back to original_dir
    writeCSV(original_dir, labelList)
    cv.destroyAllWindows() # Deletes any opened windows just in case

if __name__ == "__main__":
    main()
