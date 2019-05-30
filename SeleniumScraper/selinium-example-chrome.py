
# NOTE: Installation Instructions
# Use "pip install selenium" to install selenium for python
# Go to https://addons.mozilla.org/en-US/firefox/addon/selenium-ide/ to install the Selenium add-on for Firefox
# Go to https://github.com/mozilla/geckodriver/releases to install geckodriver
# Make sure to choose the geckodriver file that corresponds to your OS
# More helpful info at https://selenium-python.readthedocs.io/getting-started.html

import time
try:
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.common.by import By
except:
    print("please run 'pip install selenium' and try again")


opts = Options()
#opts.add_argument("—headless")     #Runs Firefox invisibly w/o a user interfaces
#opts.add_argument("-incognito")     #Rusn Firefox in incognito mode
driver = webdriver.Firefox(executable_path='/Users/summer19/Documents/geckodriver', options=opts)

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


