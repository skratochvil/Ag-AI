# -*- coding:utf-8 -*-

from flask import Flask  
from flask import render_template, flash, redirect, url_for, session
from app import app
from app.DataPreprocessing import DataPreprocessing
from app.ML_Class import Active_ML_Model
from app.SamplingMethods import lowestPercentage
from app.forms import LabelForm
from flask_bootstrap import Bootstrap
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os

bootstrap = Bootstrap(app)

@app.route("/", methods=['GET']) 
@app.route("/index",methods=['GET'])
def home():
    session.pop('model', None)
    return render_template('index.html')

@app.route("/label.html",methods=['GET', 'Post'])   
def label():
    form = LabelForm()
    if 'model' not in session:#Start
        session['confidence'] = .6
        session['accuracy'] = 0
        session['labels'] = []
        preprocess = DataPreprocessing(True)
        ml_classifier = RandomForestClassifier()
        file_name = os.path.join(app.root_path, '', 'csvOut.csv')
        data = pd.read_csv(file_name, index_col = 0, header = None)
        data = data.iloc[:, :-1]
        session['model'] = Active_ML_Model(data, ml_classifier, preprocess)
        session['queue'] = list(session['model'].sample.index)
        return render_template(url_for('label'),form = form, picture = session['queue'].pop(), confidence = session['confidence'])

    elif form.is_submitted() and session['queue'] == [] and session['labels'] == []: # Need more pictures
        sampling_method = lowestPercentage
        session['model'].Continue(sampling_method)
        session['queue'] = list(session['model'].sample.index)

    elif form.is_submitted() and session['queue'] == []:# Finished Labeling
        import numpy as np
        session['labels'].append(form._____.choice)
        session['model'].sample['y_value'] = session['labels']
        session['model'].Train(session['model'].sample)
        session['accuracy'] = np.mean(session['model'].ml_model.K_fold())
        session['labels'] = []
        if session['accuracy'] < session['confidence']:
            """
            -find variables for the intermediate results.
            -pass these labels like 'confidence'
            -change 'intermediate.html' to accept the new labels. Exp = {{ confidence }}
            """
            correct_pic, incorrect_pic = session['model'].infoForProgress()
            return render_template('intermediate.html', form = form, confidence = session['accuracy'])
        else:
            """
            -find variables for the intermediate results.
            -pass these labels like 'confidence'
            -change 'intermediate.html' to accept the new labels. Exp = {{ confidence }}
            """
            correct_pic, incorrect_pic,health_pic, blight_pic = session['model'].infoForResults()
            return render_template('final.html', form = form, confidence = session['accuracy'])

    elif form.is_submitted() and session['queue'] != []: #Still gathering labels
        session['labels'].append(form.____.data)
        return render_template(url_for('label'), picture = session['queue'].pop(), confidence = session['confidence'])
    return render_template('label.html', form = form)

@app.route("/intermediate.html",methods=['GET'])   
def intermediate():
    return render_template('intermediate.html')


@app.route("/final.html",methods=['GET'])   
def Final():
    return render_template('final.html')
#1st arg must be set to 0.0.0.0 for external server
#why port 666? 
app.run( host='127.0.0.1', port=5000, debug='True')