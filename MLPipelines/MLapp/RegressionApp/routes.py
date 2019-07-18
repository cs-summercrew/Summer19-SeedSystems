from flask import render_template, flash, redirect, url_for, request
from RegressionApp import app
from RegressionApp.forms import LoginForm
from RegressionApp import mpgPredict

# Pig latin page, when we click translate, moves to text result page
@app.route('/reg',methods=['GET','POST'])
def text():
    if request.method == 'POST':
        myinput = request.form['myinput']
        output = mpgPredict.predict(myinput)
        return render_template('regResults.html', myinput=myinput, output=output)
    return render_template('reg.html', title='Home')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        flash('Login requested for user {}, remember_me={}'.format(
            form.username.data, form.remember_me.data))
        return redirect(url_for('index'))
    return render_template('login.html', title='Sign In', form=form)

@app.route('/')
@app.route('/index')
def index():
    content = [
        {
            'topic': 'For Regression on MPG',
            'body': 'See...'
        },
        {
            'topic': 'For Classification on several Datasets',
            'body': 'See...'
        }
    ]
    return render_template('index.html', title='Home', content=content)