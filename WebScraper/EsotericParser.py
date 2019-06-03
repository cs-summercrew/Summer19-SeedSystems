
# BeautifulSoup documentation:
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/
# NOTE: Make sure that you have a parser installed, this program uses lxml,
#       See https://www.crummy.com/software/BeautifulSoup/bs4/doc/#installing-a-parser for more information

try: 
    from bs4 import BeautifulSoup
except:
    print("Please install the BeautifulSoup library and try again")
try:
    import requests
except:
    print("Please install the requests library and try again")
try:
    import os
except:
    print("Please install the os library and try again")
try:
    import shutil
except:
    print("Please install the shutil library and try again")

import os.path
import csv
import re


# If you get any of these except errors, 
# installing Anaconda should ensure that 
# you have all the proper libraries installed...
# https://www.anaconda.com/distribution/


# Global Variables
URL = "https://esolangs.org/wiki/Language_list"
baseURL = "https://esolangs.org"


def writeCSV(path, data):
    " Takes our data and writes it to a csv file"
    path = os.path.join(path, "EsoData.csv")
    with open(path, 'w') as myCSVfile:
        filewriter = csv.writer(myCSVfile, delimiter='\n', quoting=csv.QUOTE_NONE, escapechar='\\')
        # for i in range(0,len(datalist)):
            # print(datalist[i])
            # filewriter.writerow([datalist[i][0] + ',' + datalist[i][1] + ',' + datalist[i][2]])
    return path

def parseData(path):
    "Parses file directory"
    AllFiles = list(os.walk(path))[0][2]
    data = []
    for file in AllFiles:
        fpath = os.path.join(path, file)
        with open(fpath) as f:
            soup = BeautifulSoup(f, "lxml")
            #print(soup.get_text())
            catList = []
            try:
                tags = soup.find(class_="mw-normal-catlinks").ul.contents
                for tag in tags:
                    category = str(tag.contents[0].string)
                    catList.append(category)
                    # NOTE: If you don't convert a navigable string object with str, the original will carry around
                    #       a very memory intensive copy of the entire tree in the soup variable
            except:
                print("ERROR: Page doesn't have any Categories")
                catList.append("N/A")
            data.append([soup.title.string[:-10], catList])
    return data


def main():
    print("Start of main()\n")
    currDir = os.getcwd()
    path = os.path.join(currDir, "Files_To_Parse")
    data = parseData(path)
    print("\n",data)
    #writeCSV(path, data)

    print("\nEnd of main()")

if __name__ == "__main__":
    main()