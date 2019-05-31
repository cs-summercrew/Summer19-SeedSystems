
# NOTE: Installation Instructions for Firefox
#       Use "pip install selenium" to install selenium for python
#       Go to https://addons.mozilla.org/en-US/firefox/addon/selenium-ide/ to install the Selenium add-on for Firefox
#       Go to https://github.com/mozilla/geckodriver/releases to install geckodriver
#       Make sure to choose the geckodriver file that corresponds to your OS
#       Make sure that you update the path to geckodriver in main()
#       More helpful documentation info at https://selenium-python.readthedocs.io/getting-started.html

import os
import time
# NOTE: We use sleep from the time library because the code often runs faster than pages can load.
#       We also added a few extra so that you can watch what is happening.
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
    opts = Options()
    # NOTE: Incognito mode may prevent popups from appearing
    # opts.add_argument("—headless")     #Runs Firefox invisibly w/o a user interfaces
    # opts.add_argument("-incognito")     #Rusn Firefox in incognito mode
    driver = webdriver.Firefox(executable_path=path, options=opts)
    driver.get(url)
    return driver

def savepage(driver):
    "Saves the current html page as an html file"
    time.sleep(1.0)
    html = driver.page_source
    ourdir = os.getcwd()
    path = os.path.join(ourdir, "DuckDuckResults.html")
    with open(path, 'w') as f:
            f.write(html)
    return

def screenshot(driver):
    "Takes a screenshot of the current page and save it as a png"
    driver.save_screenshot("screenshot.png")
    return

def websearch(driver):
    "Conducts two websearchs using duckduckgo"
    search_form = driver.find_element_by_id('search_form_input_homepage')
    search_form.send_keys("I'm feeling famished. A churro sounds great!")
    time.sleep(2)
    search_form.clear() # Clears the searchbar of text
    search_form.send_keys("sdklfjgblsdkfjgblsdkjfgdfgadftgafdhdfhdfdag")
    search_form.submit() # Enters the current earch

    search_form = WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.ID,"search_form_input")))
    search_form.clear()
    search_form.send_keys("esoteric programming languages churro")
    search_form.submit()
    return

def closebrowser(driver):
    "Closes the browser"
    driver.close()  # Closes a single tab
    driver.quit()   # Closes all tabs
    return

def buttonclick(driver, button_element):
    "Clicks a button"
    # NOTE: ActionChains lets you automate low level interactions like mouse movements & button presses
    actions = ActionChains(driver)
    actions.move_to_element(button_element)
    actions.click(button_element)
    actions.perform()
    return

def closepopup(driver):
    "Closes the duckduckgo add-on popup for FIREFOX only"
    try: 
        # NOTE: You can find the xpath of an element by right-clicking it in the browser
        #       and clicking inspect element. Right click on the respective html code, and
        #       you should see an option to copy the element's XPATH
        close_button = driver.find_element(By.XPATH, "/html/body/div/div[5]/a/span")
        buttonclick(driver, close_button)
    except:
        print("If the popup wasn't there, you probably didn't use firefox."+"\n"+
        "Try without incognito mode or just comment out this function if it continues."+"\n"+
        "If the popup only appeared after a search, move where this function gets called")
    return

def useform(driver):
    "Opens a feedback form for our search, and interacts with it"
    # Clicks buttons to open the form
    sendfeedback_button = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH,"/html/body/div[2]/div[4]/div[2]/div/div/a")))
    buttonclick(driver, sendfeedback_button)
    good_button = driver.find_element(By.XPATH,"/html/body/div[2]/div[4]/div[2]/div/div/div/a[1]")
    buttonclick(driver, good_button)

    # Selects the correct option from the dropdown and enters text
    select = Select(driver.find_element(By.XPATH,"/html/body/div[6]/div/div/div[1]/div/div[2]/select"))
    select.select_by_index(13)
    textbox = driver.find_element(By.XPATH,"/html/body/div[6]/div/div/div[1]/div/textarea")
    textbox.send_keys("I am testing the powers of automation with Selenium. Please ignore this feedback.")

    # Clicks the buttons to close the form and the popup that follows it
    close_button1 = driver.find_element(By.XPATH,"/html/body/div[6]/div/div/div[1]/div/a[1]")
    buttonclick(driver, close_button1)
    close_button2 = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, "/html/body/div[6]/div/div/div[2]/a")))
    buttonclick(driver, close_button2)
    backandforth(driver)
    return

def backandforth(driver):
    "The driver moves to the previous item in your search history, then moves forward an item"
    driver.back()
    driver.forward()
    return

def main():
    driver = createdriver('https://duckduckgo.com',
    '/Users/summer19/Documents/GitHub/Summer19-SeedSystems/SeleniumScraper/geckodriver')
    # NOTE: You will need to change the above path to wherever you have installed geckodriver
    closepopup(driver)
    websearch(driver)
    useform(driver)
    savepage(driver)
    screenshot(driver)
    
    # Closes the open web browser
    time.sleep(2)
    closebrowser(driver)


if __name__ == "__main__":
    main()