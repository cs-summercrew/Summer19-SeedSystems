
# BeautifulSoup documentation:
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/

# NOTE: For BeautifulSoup to work, make sure that you have a parser installed, this program uses lxml,
#       which should already be installed if you're using anaconda python.
#       See https://www.crummy.com/software/BeautifulSoup/bs4/doc/#installing-a-parser for more information.

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


def setup():
    " Makes a folder to hold our html files, and return a path to that folder"
    original_dir = os.getcwd()
    dirContents = os.listdir(original_dir)
    path = os.path.join(original_dir, "Scraped_Files")
    if "Scraped_Files" in dirContents:  # We check and delete the directory because there is an error if the directory already exists
            shutil.rmtree(path) # Removes directories regardless of if they're empty
    os.mkdir(path)
    return path

def createfiles(listOflinks, path):
    "Creates file directory"
    for link in listOflinks:
        # Make the paths for each file
        name = link.get('href')
        file_name = name[6:]+".html"    # NOTE: name[6:] works for the important links, but not all of them
        filePath = os.path.join(path, file_name)
        # Get the info that will be written to the files
        URLtoLoop = baseURL + name
        info = requests.get(URLtoLoop)
        finalInfo = info.text
        # Write the files
        print("Writing",file_name)
        with open(filePath, 'w') as f:
            f.write(finalInfo)
    return

def makesoup(Ourpath):
    "Creates a beautiful soup object that can be used to parse our webpage for links"
    result = requests.get(URL)
    pagesrc = result.text # Turns the html into a single string
    soup = BeautifulSoup(pagesrc,"lxml")
    return soup


def main():
    print("Start of main()\n")
    
    Ourpath = setup()
    soup = makesoup(Ourpath)
    
    LinkList = soup.findAll('a')

    # NOTE: Experiment & Look at different languages!
    #       Our list includes the first 16 as well as some favorites and interesting languages
    #       LinkList[34] is the first link in the list of languages: !!!
    LinkList_Subset = LinkList[34:50]
    LinkList_Subset.append(LinkList[174])
    LinkList_Subset.append(LinkList[282])
    LinkList_Subset.append(LinkList[361])
    LinkList_Subset.append(LinkList[453])
    LinkList_Subset.append(LinkList[457])
    LinkList_Subset.append(LinkList[973])
    LinkList_Subset.append(LinkList[977])
    LinkList_Subset.append(LinkList[980])
    LinkList_Subset.append(LinkList[1357])
    LinkList_Subset.append(LinkList[1435])
   
    # LinkList[1441] is the last link in the list of languages: ZZZ
    
    # TODO: Maybe add a function that checks the type of the doc, look specifically for: <!DOCTYPE html>
    createfiles(LinkList_Subset, Ourpath)

    print("End of main()\n")

if __name__ == "__main__":
    main()