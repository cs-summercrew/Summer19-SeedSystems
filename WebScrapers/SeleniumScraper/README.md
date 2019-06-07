We created two examples of using selenium, one with chrome, the other with Firefox. Installing can get pretty tricky so please follow the instructions in each file closely. Selenium supports other browsers, but their installations may be different. 

Reminder: Make sure you change the filepath in the main() function to wherever you downloaded your browser's driver or your code will not work!

selTest-Chrome.py is an earlier version of our selTest example and is missing a lot of the functionality in selTest-Firefox.py and should be treaty mostly as an example of how to run selenium using chrome. However, many of the functions in selTest-Firefox could be copied into selTest-Chrome and work just fine.

selTest-Firefox has some functions that close specific popups. If you don't see the popups when running the code for whatever, just comment out those functions.

Please do note however, that there are some functions in the Selenium library that behave differently across different browsers, so using the same code across different browsers may result in unexpected behavior.

Selenium Documentation: https://selenium-python.readthedocs.io/getting-started.html