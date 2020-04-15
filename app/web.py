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

bootstrap = Bootstrap(app)

@app.route("/", methods=['GET']) 
@app.route("/index.html",methods=['GET'])
def home():
    session.pop('model', None)
    return render_template('index.html')

@app.route("/label.html",methods=['GET', 'Post'])   
def label():
    form = LabelForm()
    if 'model' not in session:#Start
        print("Start")
        session['confidence'] = .6
        session['accuracy'] = 0
        session['labels'] = []
        preprocess = DataPreprocessing(True)
        ml_classifier = RandomForestClassifier()
        file_name = os.path.join(app.root_path, '', 'csvOut.csv')
        data = pd.read_csv(file_name, index_col = 0, header = None)
        data = data.iloc[:, :-1]
        al_model = Active_ML_Model(data, ml_classifier, preprocess)
        session['sample_idx'] = list(al_model.sample.index.values)
#        print(session['sample_idx'])
        session['test'] = list(al_model.test.index.values)
#        print(session['test'])
        session['train'] = al_model.train
#        print(session['train'])
        session['model'] = True
#        print(session['model'])
        session['queue'] = list(al_model.sample.index.values)
#        print(session['queue'])
        queue = session['queue']
        img = queue.pop()
        session['queue'] = queue
        return render_template(url_for('label'),form = form, picture = img, confidence = session['confidence'])

    elif form.is_submitted() and session['queue'] == [] and session['labels'] == []: # Need more pictures
        print("Need more pictures")
        sampling_method = lowestPercentage
        file_name = os.path.join(app.root_path, '', 'csvOut.csv')
        data = pd.read_csv(file_name, index_col = 0, header = None)
        data = data.iloc[:, :-1]
        
        train_names, train_labels = list(zip(*session['train']))
        train_set = data.loc[train_names, :]
        
        train_set['y_value'] = train_labels
        
        ml_model = ML_Model(train_set, RandomForestClassifier(), DataPreprocessing(True))
        session['sample'], session['test'] = sampling_method(ml_model, 5)
        session['queue'] = list(session['model'].sample.index.values)

    elif form.is_submitted() and session['queue'] == []:# Finished Labeling
        print("Finished Labeling")
        import numpy as np
        labels = session['labels']
        labels.append(form.choice.data)
        session['labels'] = labels
        print(session['labels'])
        session['sample'] = tuple(zip(session['sample_idx'], session['labels']))
        
        if session['train'] != None:
            session['train'] = session['train'] + session['sample']
        else:
            session['train'] = session['sample']
        print(session['train'])
        
        train_img_names, train_img_label = list(zip(*session['train']))
        print(train_img_names)
        file_name = os.path.join(app.root_path, '', 'csvOut.csv')
        data = pd.read_csv(file_name, index_col = 0, header = None)
        data = data.iloc[:, :-1]
        train_set = data.loc[train_img_names, :]
        train_set['y_value'] = train_img_label
        print(train_set)
        ml_model = ML_Model(train_set, RandomForestClassifier(), DataPreprocessing(True))        
        
        session['accuracy'] = np.mean(ml_model.K_fold())
        session['labels'] = []
        if session['accuracy'] < session['confidence']:
            print (train_img_label)
            print ("Testing a print statement")
            correct_pic, incorrect_pic = ml_model.infoForProgress(train_img_names)
            correct_len = len(correct_pic)
            incorrect_len = len(incorrect_pic)
            print(incorrect_pic)
            return render_template('intermediate.html', form = form, confidence = session['accuracy'], correct = correct_pic, incorrect = incorrect_pic, correctNum = correct_len, incorrectNum = incorrect_len)
        else:
            test_set = data.loc[session['test'], :]
            correct_pic, incorrect_pic, health_pic, blight_pic = ml_model.infoForResults(train_img_label, test_set)
            print(incorrect_pic)
            return render_template('final.html', form = form, confidence = session['accuracy'])

    elif form.is_submitted() and session['queue'] != []: #Still gathering labels
        print("Still gathering labels")
        print(len(session['queue']))
        labels = session['labels']
        labels.append(form.choice.data)
        session['labels'] = labels
        queue = session['queue']
        img = queue.pop()
        session['queue'] = queue
        return render_template(url_for('label'), form = form, picture = img, confidence = session['confidence'])
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