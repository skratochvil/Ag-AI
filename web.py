# -*- coding:utf-8 -*-


from flask import Flask  
from flask import request,render_template
from flask_bootstrap import Bootstrap
app =Flask(__name__)      
bootstrap = Bootstrap(app)

@app.route("/") 
def home(): 
    return render_template('index.html')

@app.route("/index.html",methods=['GET'])   
def index():
    return render_template('index.html')

@app.route("/label.html",methods=['GET'])   
def label():
    return render_template('label.html')

@app.route("/Intermediate.html",methods=['GET'])   
def Intermediate():
    return render_template('Intermediate.html')


@app.route("/Final.html",methods=['GET'])   
def Final():
    return render_template('Final.html')
app.run( host='127.0.0.1', port=666, debug='True')

