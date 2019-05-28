# requests
import requests
url = "https://www.cs.hmc.edu/~dodds/demo.html"
result = requests.get(url)
pagesrc = result.text

# beautiful soup
from bs4 import BeautifulSoup
soup = BeautifulSoup(pagesrc,"lxml")
List1 = soup.findAll('li')  # all list items
List2 = soup.findAll('li', class_="latte") # w/class "latte"



