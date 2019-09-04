# Authors: CS-World Domination Summer19 - DM

# BeautifulSoup documentation:
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/

# NOTE: Make sure that you have a parser installed, this program uses lxml,
#       See https://www.crummy.com/software/BeautifulSoup/bs4/doc/#installing-a-parser for more information

try: 
    from bs4 import BeautifulSoup
except:
    print("Please install the BeautifulSoup4 library and try again")
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
import string


# If you get any of these except errors, 
# installing Anaconda should ensure that 
# you have all the proper libraries installed...
# https://www.anaconda.com/distribution/


def writeCSV(path, data):
    " Takes our data and writes it to a csv file"
    path = os.path.join(path, "EsoData.csv")
    with open(path, 'w') as myCSVfile:
        print("Writing EsoData.csv")
        filewriter = csv.writer(myCSVfile, delimiter='\n', quoting=csv.QUOTE_NONE, escapechar='\\')
        for i in range(0,len(data)):
                filewriter.writerow([data[i][0] + ',' + data[i][1] + ',' + data[i][2] + ',' + data[i][3] + ',' + data[i][4] + ',' + data[i][5]])
    return

def simplifyText(myString):
    "Strips the input text of all whitespace and punctuation, and changes it to all lowercase"
    myString = myString.lower()
    myString = myString.replace(" ", "")
    translator = str.maketrans('', '', string.punctuation)
    myString = myString.translate(translator)
    return myString

def parseData(path):
    "Parses file directory"
    AllFiles = list(os.walk(path))[0][2]
    # Start: Code specific to Mac only
    if '.DS_Store' in AllFiles:
        # DS_Store files are an annoying Mac feature
        ind = AllFiles.index('.DS_Store')
        AllFiles = AllFiles[:ind] + AllFiles[ind+1:]
    # End: Code specific to Mac only
    data = []
    # Data Index Titles
    data.append(["Title", "Page Last Edited", "Page is a Stub", "Page contains Hello World", "Is Turing Complete", "Has External Resources"])

    for file in AllFiles:
        fpath = os.path.join(path, file)
        with open(fpath) as f:
            soup = BeautifulSoup(f, "lxml")

            # Finds the categories section and puts each category into catList
            # NOTE: We use try and except because if a page doesn't have the Categories section, 
            #       and thus the respective html class used below, then soup.find() will throw an error
            catList = []
            try:
                tags = soup.find(class_="mw-normal-catlinks").ul.contents
                for tag in tags:
                    category = str(tag.contents[0].string)  # Important Step!!! See NOTE below
                    catList.append(category)
                    # NOTE: .string does not return a python string! Convert it with str() or it will act like a normal string,
                    #       but carry around a memory intensive copy of the entire BeautifulSoup tree
            except:
                catList.append("N/A")
            
            # Finds each header and puts it into a list
            headList = []
            try:
                headers = soup.find_all("h2")
                for header in headers:
                    OurHeader = str(header.string)
                    headList.append(OurHeader)
                headList = headList[:-1]    # The last header is non-unique
            except:
                headList.append("N/A")

            # Finds the time of the last edit
            lastEdit_tag = soup.find(id="footer-info-lastmod")
            lastEdit = str(lastEdit_tag.string)[30:-1]
            lastEdit = lastEdit.replace(",","")
            
            # Checks if "hello world" is in the file's text
            fileText = simplifyText(str(soup.get_text()))
            containsHelloWorld = int(('helloworld' in fileText))
            
            # Checks if the page is a stub
            isStub = int(("stub" in fileText) or ("Stubs" in catList))

            # Checks if page's subject language is Turing Complete by searching catList
            isTuringComp = int(("Turing complete" in catList) or ("Turing tarpits" in catList))

            # Checks if the page has an external rescources header
            hasExtRes = int("External resources" in headList)

            data.append([soup.title.string[:-10], lastEdit, str(isStub), str(containsHelloWorld), str(isTuringComp), str(hasExtRes)])
    print("Number of files:", len(data)-1)
    return data


def main():
    print("Start of main()\n")
    currDir = os.getcwd()
    path = os.path.join(currDir, "Files_To_Parse")
    data = parseData(path)
    writeCSV(currDir, data)

    print("\nEnd of main()")

if __name__ == "__main__":
    main()