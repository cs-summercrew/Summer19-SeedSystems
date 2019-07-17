from flask import Flask
from configuration import Config

app = Flask(__name__)
app.config.from_object(Config)

from RegressionApp import routes