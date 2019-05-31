

# NOTE: Installation Instructions
# Use "pip install selenium" to install selenium for python
# Go to the chrome web store to install the Selenium add-on for chrome (published by seliniumhq.org)
# Go to https://sites.google.com/a/chromium.org/chromedriver/downloads to install chromedriver
# Make sure to choose the chromedriver file that corresponds to your browser (check at whatismybrowser.com)
# More helpful info at https://selenium-python.readthedocs.io/getting-started.html

import time
try:
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait 
    from selenium.webdriver.support import expected_conditions as EC 
    from selenium.common.exceptions import TimeoutException
except:
    print("please run 'pip install selenium' and try again")


<<<<<<< HEAD
opts = Options()
#opts.add_argument("—headless")     #Runs invisibly w/o a user interfaces
#opts.add_argument("-incognito")     #Runs in incognito mode
driver = webdriver.Chrome(executable_path='/Users/summer19/Downloads/chromedriver', options=opts)

driver.get('https://duckduckgo.com')
html1 = driver.page_source  #Gets the html of a webpage
# Puts text into the duckduckGo search bar, and enters a search
search_form = driver.find_element_by_id('search_form_input_homepage')
search_form.send_keys("I'm feeling famished. A churro sounds great!")
time.sleep(5)
search_form.clear() # Clears 
search_form.send_keys("esoteric programming languages")
search_form.submit()
time.sleep(2)
assert "No results found" not in driver.page_source
#driver.save_screenshot("screenshot.png")    #Takes a screenshot
link = driver.find_element_by_link_text('Esoteric programming language - Esolang')
print(link.text)


# # Prints all the links on the webpage below
# driver.get('https://www.w3.org/')
# for a in driver.find_elements_by_xpath('.//a'):
#     print(a.get_attribute('href'))


time.sleep(5)
driver.close()
driver.quit()
=======
def createdriver(url, path):
    "Creates the selenium driver object"
    opts = Options()
    #opts.add_argument("—headless")     #Runs Firefox invisibly w/o a user interfaces
    #opts.add_argument("-incognito")     #Rusn Firefox in incognito mode
    driver = webdriver.Firefox(executable_path=path, options=opts)
    driver.get(url)
    return driver
>>>>>>> master

def savepage(driver):
    "Saves the current html page as an html file"
    html = driver.page_source
    return html

def screenshot(driver):
    "Takes a screenshot of the current page and save it as a png"
    #driver.save_screenshot("screenshot.png")    #Takes a screenshot
    return

def websearch(driver):
    "Showcases a websearch with duckduckgo"
    search_form = driver.find_element_by_id('search_form_input_homepage')
    search_form.send_keys("I'm feeling famished. A churro sounds great!")
    time.sleep(5)
    search_form.clear() # Clears 
    search_form.send_keys("esoteric programming languages churro")
    search_form.submit()
    time.sleep(2)
    assert "No results found" not in driver.page_source
    return

def main():
    """ run this file as a script """
    # # Prints all the links on the webpage below
    # driver.get('https://www.w3.org/')
    # for a in driver.find_elements_by_xpath('.//a'):
    #     print(a.get_attribute('href'))
    
    driver = createdriver('https://duckduckgo.com',
    '/Users/summer19/Documents/GitHub/Summer19-SeedSystems/SeleniumScraper/geckodriver')
    savepage(driver)
    screenshot(driver)

    # link = driver.find_element_by_link_text('Esoteric programming language - Esolang')
    # print(link.text)
    # print(link.get_attribute('href'))
    
    # Closes the open web browser
    time.sleep(5)
    driver.close()
    driver.quit()


if __name__ == "__main__":
    main()