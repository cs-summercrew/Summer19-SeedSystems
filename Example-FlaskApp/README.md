## Basic Flask Application:

We have provided all of the code to deploy an example Flask application. The
application contained within this folder provides the functionality for a 
pig latin translator as well as an image inverter. The purpose of this example
app is to demonstrate the inner-working of a flask app, providing a reference
for how the different files in the app interact with one another. In order to 
run the application, follow the instructions detailed below. We also provide
instructions for how to deploy your webapp live on the internet via a temporary
domain using software called ngrok.

---

![Program Image](https://github.com/cs-summercew/Summer19-SeedSystems/blob/master/ExampleFlaskApp/tree.png)

Initializes the configurations of our app


Sets the static folder as the default upload folder for files

#### Running the App Locally:
Note that Steps 1-3 deal with running your flask app from a virtual environment with virtualenv. Many tutorials online use virtualenv, but just know that these steps are entirely optional.  

1. Step into the ExampleApp directory

2. If you have not done so, use the command:

```pip install virtualenv``` 

If above command is not working, try:

```pip3 install virtualenv``` 

Then create a new virtual environment using the following command: (second venv is for the name of folder)

```python3 -m venv venv```

This will create a new virtual environment in a folder called venv.


3. To run your virtual enviroment use the following command:

```
for Mac/Linux Users:    source venv/bin/activate

for Windows:            venv\Scripts\activate 
```

For any trouble with virtual environments, visit the website below

* [Virtual Environment Troubleshooting](https://docs.python-guide.org/dev/virtualenvs/)


4. Run the following commands to ensure that all of the necessary libraries 
   have been downloaded

```
pip (or conda) install Flask twilio
pip (or conda) install Pillow
```

For help with installing the necessary libraries, visit the following sites

* [Setting up Python and Flask Environment](https://www.twilio.com/docs/usage/tutorials/how-to-set-up-your-python-and-flask-development-environment)

* [Installing Pillow Library](https://pillow.readthedocs.io/en/stable/installation.html)

5. Run the following commands 

For Mac:

```
export FLASK_APP=example-app.py
flask run
```

For Windows:

```
set FLASK_APP=example-app.py
flask run
```
        
6. Visit the website:

http://localhost:5000/

The webapp should be active and functional if you followed the above steps correctly.


#### Ngrok Deployment:

1. Download the ngrok binary executable from the provided website and move it to the ExampleApp directory. 
   Instructions for installation are available on the website:

* [Install ngrok](https://ngrok.com/download)

2. Make sure your Flask app is running, and open a separate terminal. In the same directory as the ngrok
   executable, run the command

```./ngrok http 5000```

3. Visit the website providing under forwarding header. This will take you to your webapp, which
   is now accessible from any machine so long as the ngrok session is running.


#### Heroku Deployment:

1. Download and install the Heroku CLI from the following website:

* [Install Heroku CLI](https://devcenter.heroku.com/articles/getting-started-with-python#set-up)

2. Navigate to the ExampleApp directory and login in to Heroku CLI in terminal using the command

```heroku login```

3. Now initialize a virtual environment if you haven't already using the commands detailed above.
   Install the various the various libraries used by the app, as well as a new one called gunicorn.
   
```
pip install Flask twilio
pip install Pillow`
pip install gunicorn
```

4. Save these as required libraries to a file called requirements.txt using the following command:

```pip freeze > requirements.txt```

5. Create another file called "Procfile" (no extension). In this file, put the following code:

```web: gunicorn example-app:app```
        
In the place of example-app, put the name of your application if it is different

6. Initialize a git repo in heroku using the command git init while in the directory with your app.
        
 `git init`
        
Push all of the files onto this git repo using the commands:
       
 ```
 git add .
 git commit -m "Initial Commit"
 git push heroku master
 ```

7. Finally deploy the web app and open it:

```
heroku ps:scale web=1
heroku open
```


If you would like to develop your own Flask App from scratch, be sure to check out this tutorial. 
It provides an in depth tutorial that covers everything one might need to get started in creating
web apps. 

* [Intro Flask Tutorial](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world)

