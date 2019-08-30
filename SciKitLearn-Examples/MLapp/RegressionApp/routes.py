# Authors: CS-World Domination Summer19 - DM
from flask import render_template, flash, redirect, url_for, request
from RegressionApp import app
from RegressionApp import mpgPredict
from RegressionApp import survPredict

# Classification page, when submit is clicked, it moves to the clsResults page
@app.route('/cls',methods=['GET','POST'])
def cls():
    if request.method == 'POST':
        clsInput = request.form['clsInput']
        clsOutput = survPredict.predict(clsInput)
        return render_template('clsResults.html', clsInput=clsInput, clsOutput=clsOutput)
    return render_template('cls.html', title='Titanic')

# Regression page, when submit is clicked, it moves to the regResults page
@app.route('/reg',methods=['GET','POST'])
def reg():
    if request.method == 'POST':
        myinput = request.form['myinput']
        output = mpgPredict.predict(myinput)
        return render_template('regResults.html', myinput=myinput, output=output)
    return render_template('reg.html', title='MPG')

@app.route('/')
@app.route('/index')
def index():
    content = [
        {
            'topic': 'Click the Regression tab to see the MPG predictor.',
        },
        {
            'topic': 'Click the Classification tab to see the Titanic Survival predictor.',
        }
    ]
    return render_template('index.html', title='Home', content=content)