
# NOTE: Installation Instructions for Firefox
# Use "pip install selenium" to install selenium for python
# Go to https://addons.mozilla.org/en-US/firefox/addon/selenium-ide/ to install the Selenium add-on for Firefox
# Go to https://github.com/mozilla/geckodriver/releases to install geckodriver
# Make sure to choose the geckodriver file that corresponds to your OS
# More helpful documentation info at https://selenium-python.readthedocs.io/getting-started.html

import time
try:
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.common.by import By
except:
    print("please run 'pip install selenium' and try again")


def createdriver(url, path):
    "Creates the selenium driver object"
    opts = Options()
    #opts.add_argument("—headless")     #Runs Firefox invisibly w/o a user interfaces
    #opts.add_argument("-incognito")     #Rusn Firefox in incognito mode
    driver = webdriver.Firefox(executable_path=path, options=opts)
    driver.get(url)
    return driver

def savepage(driver):
    "Saves the current html page as an html file"
    html = driver.page_source
    with open(".", 'w') as f:
        f.write(html)
    return

def screenshot(driver):
    "Takes a screenshot of the current page and save it as a png"
    driver.save_screenshot("screenshot.png")
    return

def websearch(driver):
    "Showcases a websearch with duckduckgo"
    search_form = driver.find_element_by_id('search_form_input_homepage')
    search_form.send_keys("I'm feeling famished. A churro sounds great!")
    time.sleep(2)
    search_form.clear() # Clears the searchbar of text
    search_form.send_keys("sdklfjgblsdkfjgblsdkjfgdfgadftgafdhdfhdfdag")
    search_form.submit() # Enters the current earch

    # # If a page or some elements of it load slowly, then you can wait for
    # # it to load with code like below
    # try:
    #     print('hi')
    #     # search_form = driver.find_element_by_id('search_form_input')
    #     # print(search_form)
    #     wait = WebDriverWait(driver, 3)
    #     element = wait.until(EC.visibility_of((By.ID, "'search_form_input'")))
    #     print('hi')
    #     print(element)
    #     #time.sleep(3)
    # except:
    #     print("Your Internet must suck! The webpage took too long to load.")
    
    #NOTE: The code runs faster than the page can load, so you need to make it wait
    time.sleep(2)
    search_form = driver.find_element_by_id('search_form_input')
    search_form.clear()
    search_form.send_keys("esoteric programming languages churro")
    search_form.submit()
    
    # Checks that we're on the second search, and have results
    time.sleep(0.1)
    assert "No results found" not in driver.page_source
    
    return

def closebrowser(driver):
    "Closes the browser"
    driver.close()  # Closes a single tab
    driver.quit()   # Closes all tabs
    return

def main():
    """ run this file as a script """
    # # Prints all the links on the webpage below
    # driver.get('https://www.w3.org/')
    # for a in driver.find_elements_by_xpath('.//a'):
    #     print(a.get_attribute('href'))
    
    driver = createdriver('https://duckduckgo.com',
    '/Users/summer19/Documents/GitHub/Summer19-SeedSystems/SeleniumScraper/geckodriver')
    websearch(driver)
    savepage(driver)
    #screenshot(driver)

    # link = driver.find_element_by_link_text('Esoteric programming language - Esolang')
    # print(link.text)
    # print(link.get_attribute('href'))
    
    # Closes the open web browser
    time.sleep(2)
    closebrowser(driver)



if __name__ == "__main__":
    main()