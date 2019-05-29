
# beautiful soup documentation:
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/
from bs4 import BeautifulSoup

import requests

import os
import os.path
import shutil
url = "https://esolangs.org/wiki/Language_list"
baseURL = "https://esolangs.org"

#TODO: I remember Dodds mentioning that we should have some trys & excepts for our imports w/ error messages
# that mention that the user needs to install the appropriate packages...

def setup():
    " Makes a folder to hold our html files"
    original_dir = os.getcwd()
    dirContents = os.listdir(original_dir)
    path = os.path.join(original_dir, "Scraped_Files")
    if "Scraped_Files" in dirContents:
            shutil.rmtree(path) #Removes directories regardless of if they're empty
    os.mkdir(path)
    return path

def createfiles(listOflinks, path):
    "Creates file directory"
    for link in listOflinks:
        # Make the paths for each file
        name = link.get('href')
        print(name)
        file_name = name[6:]+".html"    #Will need to check that this works for all links on larger lists
        print("File name is",file_name)
        filePath = os.path.join(path, file_name)
        # Get the info that will be written to the files
        URLtoLoop = baseURL + name
        print(URLtoLoop)
        info = requests.get(URLtoLoop)
        finalInfo = info.text
        # Write the files
        with open(filePath, 'w') as f:
            f.write(finalInfo)
    return

def scrape(Ourpath):
    result = requests.get(url)
    pagesrc = result.text #returns a string

    soup = BeautifulSoup(pagesrc,"lxml")
    LinkList = soup.findAll('a')
    LinkList_Subset=LinkList[35:40]
    createfiles(LinkList_Subset, Ourpath)


def main():
    print("Start of main()\n")
  
    #print(LinkList[33]) #external resources link
    #Links to other lang pages start @35 (!!!)
    
    Ourpath = setup()
    scrape(Ourpath)
    

    print("End of main()\n")

if __name__ == "__main__":
    main()