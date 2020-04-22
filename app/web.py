# -*- coding:utf-8 -*-

from flask import Flask  
from flask import render_template, flash, redirect, url_for, session
from app import app
from app.DataPreprocessing import DataPreprocessing
from app.ML_Class import Active_ML_Model, AL_Encoder, ML_Model
from app.SamplingMethods import lowestPercentage
from app.forms import LabelForm
from flask_bootstrap import Bootstrap
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
import numpy as np
import boto3
from io import StringIO

bootstrap = Bootstrap(app)

def getData():
    """
    Gets and returns the csvOut.csv as a DataFrame.
    
    Returns
    -------
    data : Pandas DataFrame
        The data that contains the features for each image.
    """
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket = 'cornimagesbucket', Key = 'csvOut.csv')
    body = obj['Body']
    csv_string = body.read().decode('utf-8')
    data = pd.read_csv(StringIO(csv_string))
    return data.iloc[:, :-1]

def createMLModel(data):
    """
    Prepares the training set and creates a machine learning model using the training set.
    
    Parameters
    ----------
    data : Pandas DataFrame
        The data that contains the features for each image
    
    Returns
    -------
    ml_model : ML_Model class object
        ml_model created from the training set.
    train_img_names : String
        The names of the images.
    """
    train_img_names, train_img_label = list(zip(*session['train']))
    train_set = data.loc[train_img_names, :]
    train_set['y_value'] = train_img_label
    ml_model = ML_Model(train_set, RandomForestClassifier(), DataPreprocessing(True))
    return ml_model, train_img_names

def renderLabel(form):
    """
    prepairs a render_template to show the label.html web page.
    
    Parameters
    ----------
    form : LabelForm class object
        form to be used when displaying label.html
    
    Returns
    -------
    render_template : flask function
        renders the label.html webpage.
    """
    queue = session['queue']
    img = queue.pop()
    session['queue'] = queue
    return render_template(url_for('label'), form = form, picture = img, confidence = session['confidence'])
    
def initializeAL(form, confidence_break = .7):
    """
    Initializes the active learning model and sets up the webpage with everything needed to run the application.
    
    Parameters
    ----------
    form : LabelForm class object
        form to be used when displaying label.html
    confidence_break : number
        How confident the model is.
    
    Returns
    -------
    render_template : flask function
        renders the label.html webpage.
    """
    preprocess = DataPreprocessing(True)
    ml_classifier = RandomForestClassifier()
    data = getData()
    al_model = Active_ML_Model(data, ml_classifier, preprocess)
    
    session['confidence'] = 0
    session['confidence_break'] = confidence_break
    session['labels'] = []
    session['sample_idx'] = list(al_model.sample.index.values)
    session['test'] = list(al_model.test.index.values)
    session['train'] = al_model.train
    session['model'] = True
    session['queue'] = list(al_model.sample.index.values)
    
    return renderLabel(form)
    
def getNextSetOfImages(form, sampling_method):
    """
    Uses a sampling method to get the next set of images needed to be labeled.
    
    Parameters
    ----------
    form : LabelForm class object
        form to be used when displaying label.html
    sampling_method : SamplingMethods Function
        function that returns the queue and the new test set that does not contain the queue.
    
    Returns
    -------
    render_template : flask function
        renders the label.html webpage.
    """
    data = getData()
    ml_model, train_img_names = createMLModel(data)
    test_set = data[data.index.isin(train_img_names) == False]
    
    session['sample_idx'], session['test'] = sampling_method(ml_model, test_set, 5)
    session['queue'] = session['sample_idx'].copy()

    return renderLabel(form)

def prepairResults(form):
    """
    Creates the new machine learning model and gets the confidence of the machine learning model.
    
    Parameters
    ----------
    form : LabelForm class object
        form to be used when displaying label.html
    
    Returns
    -------
    render_template : flask function
        renders the appropriate webpage based on new confidence score.
    """
    session['labels'].append(form.choice.data)
    session['sample'] = tuple(zip(session['sample_idx'], session['labels']))
    
    if session['train'] != None:
        session['train'] = session['train'] + session['sample']
    else:
        session['train'] = session['sample']

    data = getData()
    ml_model, train_img_names = createMLModel(data)    
    
    session['confidence'] = np.mean(ml_model.K_fold())
    session['labels'] = []
    
    if session['confidence'] < session['confidence_break']:
        correct_pic, incorrect_pic = ml_model.infoForProgress(train_img_names)
        return render_template('intermediate.html', form = form, confidence = session['confidence'], correct = correct_pic, incorrect = incorrect_pic, correctNum = len(correct_pic), incorrectNum = len(incorrect_pic))
    else:
        test_set = data.loc[session['test'], :]
        correct_pic, incorrect_pic, health_pic, blight_pic = ml_model.infoForResults(train_img_names, test_set)
        return render_template('final.html', form = form, confidence = session['confidence'], correct = correct_pic, incorrect = incorrect_pic, correctNum = len(correct_pic), incorrectNum = len(incorrect_pic), healthy = health_pic, unhealthy = blight_pic, healthyNum = len(health_pic), unhealthyNum = len(blight_pic))

@app.route("/", methods=['GET']) 
@app.route("/index.html",methods=['GET'])
def home():
    session.pop('model', None)
    return render_template('index.html')

@app.route("/label.html",methods=['GET', 'Post'])   
def label():
    form = LabelForm()
    if 'model' not in session:#Start
        return initializeAL(form, .7)

    elif session['queue'] == [] and session['labels'] == []: # Need more pictures
        return getNextSetOfImages(form, lowestPercentage)

    elif form.is_submitted() and session['queue'] == []:# Finished Labeling
        return prepairResults(form)
        
    elif form.is_submitted() and session['queue'] != []: #Still gathering labels
        session['labels'].append(form.choice.data)
        return renderLabel(form)
    
    return render_template('label.html', form = form)

@app.route("/intermediate.html",methods=['GET'])   
def intermediate():
    return render_template('intermediate.html')

@app.route("/final.html",methods=['GET'])   
def Final():
    return render_template('final.html')
#1st arg must be set to 0.0.0.0 for external server
#why port 666? 
#app.run( host='127.0.0.1', port=5000, debug='True', use_reloader = False)

