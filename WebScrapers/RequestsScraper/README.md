EsotericScraper.py uses the requests library to scrape the html from links at https://esolangs.org/wiki/Language_list.
EsotericScraper.py creates and outputs to the file Scraped_Files.

Files_To_Parse is an exact copy of Scraped_Files, and is used as input for EsotericParser.py.

EsotericParser.py parses the html files in Files_To_Parse using the BeautifulSoup Library, and outputs EsoData.csv

EsoData.csv contains summary data for each html file in Files_To_Parse
