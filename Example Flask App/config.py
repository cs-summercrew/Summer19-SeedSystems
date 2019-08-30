# Authors: CS-World Domination Summer19 - JG
import os

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    # Set the upload folder as the static folder in app directory
    UPLOAD_FOLDER = 'app/static'