
# BeautifulSoup documentation:
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/
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

# If you get any of these except errors, 
# installing Anaconda should ensure that 
# you have all the proper libraries installed...
# https://www.anaconda.com/distribution/


# Global Variables
URL = "https://esolangs.org/wiki/Language_list"
baseURL = "https://esolangs.org"


def writeCSV(currDir, data):
    " Makes a folder to hold our html files, and return a path to that folder"
    path = os.path.join(currDir, "Files_To_Parse")
    path = os.path.join(path, "EsoData.csv")
    with open(path, 'w') as myCSVfile:
        filewriter = csv.writer(myCSVfile, delimiter='\n', quoting=csv.QUOTE_NONE, escapechar='\\')
        # for i in range(0,len(datalist)):
            # print(datalist[i])
            # filewriter.writerow([datalist[i][0] + ',' + datalist[i][1] + ',' + datalist[i][2]])
    return path

def parseData(currDir):
    "Creates file directory"
    
    return "nothing"


def main():
    print("Start of main()\n")
    currDir = os.getcwd()
    data = parseData()
    writeCSV(currDir, data)

    print("End of main()\n")

if __name__ == "__main__":
    main()