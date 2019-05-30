#Basic web scraper example

#!!!!!!!!!!!!!!!!!!!!!!!!!!
#TODO:   DOWNLOAD chrome webdriver FIRST!!!!!!!!!
#     https://sites.google.com/a/chromium.org/chromedriver/downloads
#If you're not sure which version you need, look at whatismybrowser.com
#MAKE SURE YOU DOWNLOAD THE VERSION YOUR BROWSER WANTS
#</!!!!!!!!!!!!!!!!!!!!!!>

#try to import the libraries we need. Make sure you have 
# TODO installed selenium in python (pip install selenium)
# TODO AND as an extension in Chrome. Download selenium from the 
# Chrome web store (https://chrome.google.com/webstore/search/selenium) 
# (search for selipublished by seleniumhq.org)
try:
    from selenium import webdriver 
    from selenium.webdriver.common.by import By 
    from selenium.webdriver.support.ui import WebDriverWait 
    from selenium.webdriver.support import expected_conditions as EC 
    from selenium.common.exceptions import TimeoutException
except:
    print("please run 'pip install selenium' and try again")


def main():
    
    #sets chrome to incognito mode. Remove if desired
    option = webdriver.ChromeOptions()
    option.add_argument("—incognito")

    #TODO: change executable_path to wherever you downloaded chromedriver to
    browser = webdriver.Chrome(executable_path='/Users/summer19/Downloads/chromedriver', chrome_options=option)

    #pick a site to scrape
    browser.get("https://github.com/TheDancerCodes")

    # # Wait 20 seconds for page to load
    # timeout = 20
    # try: #this command waits until the browser times out at 20 seconds OR the 
    #     #profile picture loads(’avatar width-full rounded-2')
    #     WebDriverWait(browser, timeout).until(EC.visibility_of_element_located((By.XPATH, "//img[@class=’avatar width-full rounded-2']")))
    # except TimeoutException:
    #     print("Timed out waiting for page to load")
    #     browser.quit()

    # # find_elements_by_xpath returns an array of selenium objects.
    # titles_element = browser.find_elements_by_xpath("//a[@class='text-bold']")
    # # create a list of actual title, not selenium objects
    # titles = [x.text for x in titles_element]
    # # print out all the titles.
    # print('titles:')
    # print(titles, '\n')

    # #now get the languages of each repo
    # language_element = browser.find_elements_by_xpath("//p[@class='mb-0 f6 text-gray']")
    # # same concept as for list-comprehension above.
    # languages = [x.text for x in language_element]
    # print("languages:")
    # print(languages, '\n')

    # #zip will combine our two lists into a tuple, then we can print them
    # for title, language in zip(titles, languages):
    #     print("RepoName : Language")
    #     print(title + ": " + language, '\n')


if __name__ == "__main__":
    main()


