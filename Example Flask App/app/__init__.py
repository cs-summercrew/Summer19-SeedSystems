# Authors: CS-World Domination Summer19 - JG
try:
    from flask import Flask
except:
    print('Make sure to pip install Flask twilio')
from config import Config

app = Flask(__name__, static_url_path='')
app.config.from_object(Config)

from app import routes