
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


def setup():
    " Makes a folder to hold our html files, and return a path to that folder"
    original_dir = os.getcwd()
    dirContents = os.listdir(original_dir)
    path = os.path.join(original_dir, "Scraped_Files_Example")
    return path

def createfiles(listOflinks, path):
    "Creates file directory"
    
    return


def main():
    print("Start of main()\n")
    
    setup()

    print("End of main()\n")

if __name__ == "__main__":
    main()