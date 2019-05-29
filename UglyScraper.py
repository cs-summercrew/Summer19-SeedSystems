
# beautiful soup documentation:
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/
from bs4 import BeautifulSoup

import requests

import os
import os.path
import shutil

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
    # NOTE: Need to either copy files into the Scrapped_Files folder
    # or somehow directly create the files in that folder
    
    for link in listOflinks:
        name = link.get('href')
        print(name)
        file_name = name[6:]+".html"    #Will need to check that this works for all links on larger lists
        print("File name is",file_name)
        filePath = os.path.join(path, file_name)
        #os.mkdir(filePath)
        with open(filePath, 'w') as f:
            f.write('')
    return

def main():
    print("Start of main()\n")
    
    url = "https://esolangs.org/wiki/Language_list"
    baseURL = "https://esolangs.org"
    result = requests.get(url)
    pagesrc = result.text #returns a string

    soup = BeautifulSoup(pagesrc,"lxml")
    LinkList = soup.findAll('a')

    #print(LinkList[33]) #external resources link
    #Links to other lang pages start @35 (!!!)
    
    Ourpath = setup()
    LinkList_Subset = LinkList[35:40]
    createfiles(LinkList_Subset, Ourpath)

    print("End of main()\n")

if __name__ == "__main__":
    main()