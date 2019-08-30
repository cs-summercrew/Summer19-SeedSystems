# NOTE: Installation Instructions for Firefox
#       Use "pip install selenium" to install selenium for python
#       Go to https://addons.mozilla.org/en-US/firefox/addon/selenium-ide/ to install the Selenium add-on for Firefox
#       Go to https://github.com/mozilla/geckodriver/releases to install geckodriver
#       Make sure to choose the geckodriver download that corresponds to your OS
#       Make sure that you update the path to geckodriver in the main() function
#       More helpful documentation info at https://selenium-python.readthedocs.io/getting-started.html

import os
import time
# NOTE: We used .sleep() from the time library to make it easier to watch what is happening (It is not necessary for the scraper)
loop = True
barcode = ""
try:
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.support.ui import Select
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
except:
    print("please run 'pip install selenium' and try again")

def createdriver(url, path):
    "Creates the selenium driver object"
    driver = webdriver.Firefox(executable_path=path)
    driver.get(url)
    return driver

def screenshot(driver):
    "Takes a screenshot of the current page and save it as a png"
    driver.save_screenshot("screenshot.png")
    return

def websearch(driver,code):
    "Conducts two websearchs using duckduckgo"
    search_form = driver.find_element_by_id('search-input')
    search_form.send_keys(code)
    search_form.submit() # Enters the current earch
    time.sleep(2)
    return

def closebrowser(driver):
    "Closes the browser"
    driver.close()  # Closes a single tab
    driver.quit()   # Closes all tabs
    return

def main():
    driver = createdriver('https://www.barcodelookup.com',
    '/Users/summer19/Documents/GitHub/Summer19-SeedSystems/WebScrapers/SeleniumScraper/geckodriver')
    # NOTE: You will need to change the above path to wherever you have installed geckodriver
    barcode = 0
    while(loop):
        if barcode != 0:
            websearch(driver, barcode)
            barcode = 0
        else:
            barcode = input()

    
    # Closes the open web browser
    time.sleep(2)
    closebrowser(driver)


if __name__ == "__main__":
    main()