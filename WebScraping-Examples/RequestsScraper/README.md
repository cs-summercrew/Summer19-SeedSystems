### Python Programs:
`EsotericScraper.py` uses the requests library to scrape the html from links at https://esolangs.org/wiki/Language_list.
Make sure you first run `EsotericScraper.py` before running `EsotericParser.py` so that the necessary `Scraped_Files` folder is created.
`EsotericScraper.py` creates and outputs to the file `Scraped_Files`.  

`EsotericParser.py` parses the html files in `Scraped_Files` using the BeautifulSoup Library, and outputs `EsoData.csv`  

### Files created by the above Programs:
`Scraped_Files` is is a folder of html files created by `EsotericScraper.py`  
`EsoData.csv` contains summary data for each html file in `Scraped_Files`, and is created by `EsotericParser.py`  
