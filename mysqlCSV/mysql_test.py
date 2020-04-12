#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 22:35:48 2020

@author: lian
"""
import csv
#import pandas as pd
#import io
#import urllib
from flask import Flask
from flask_mysqldb import MySQL
from flask import jsonify

mysql_test = Flask(__name__)

mysql_test.config['MYSQL_HOST'] = 'lduan.mysql.pythonanywhere-services.com'
mysql_test.config['MYSQL_USER'] = 'lduan'
mysql_test.config['MYSQL_PASSWORD'] = 'csci4970'
mysql_test.config['MYSQL_DB'] = 'lduan$CSV'


mysql = MySQL(mysql_test)

@mysql_test.route('/')
def init():

    return "Started"

@mysql_test.route('/create')
def create():

    cursor = mysql.connection.cursor()
    cursor.execute("CREATE TABLE test" "(picNum VARCHAR(255) PRIMARY KEY, value1 FLOAT NOT NULL, value2 FLOAT NOT NULL, value3 FLOAT NOT NULL, value4 FLOAT NOT NULL,\
                                         value5 FLOAT NOT NULL, value6 FLOAT NOT NULL, value7 FLOAT NOT NULL, value8 FLOAT NOT NULL, value9 FLOAT NOT NULL,\
                                         value10 FLOAT NOT NULL, value11 FLOAT NOT NULL, value12 FLOAT NOT NULL, value13 FLOAT NOT NULL, value14 FLOAT NOT NULL, value15 FLOAT NOT NULL,\
                                         value16 FLOAT NOT NULL, value17 FLOAT NOT NULL, value18 FLOAT NOT NULL, value19 FLOAT NOT NULL, value20 FLOAT NOT NULL, value21 FLOAT NOT NULL, value22 FLOAT NOT NULL, value23 FLOAT NOT NULL, value24 INT, value25 INT, value26 INT, value27 INT, letter VARCHAR(255))")
    return "Done"

@mysql_test.route('/insert')
def insert():

    cursor = mysql.connection.cursor()
    #if cursor.is_connected():
        #print('Connected to MySQL database')

    #csv_data = pd.read_csv(url)
    #url="https://raw.githubusercontent.com/skratochvil/Ag-AI/Milestone2/app/csvOut.csv"
    #url_open = urllib.request.urlopen(url)
    #csvfile = csv.reader(io.StringIO(url_open.read().decode('utf-8')), delimiter=',')
    csv_data = csv.reader(open('/home/lduan/mysite/mysqlCSV/csvOut.csv'))
    for row in csv_data:
        if row:
            cursor.execute("INSERT IGNORE INTO test"
                       "(picNum, value1, value2, value3, value4, value5,\
                       value6, value7, value8, value9, value10, value11,\
                       value12, value13, value14, value15, value16, value17,\
                       value18, value19, value20, value21, value22, value23,\
                       value24, value25, value26, value27, letter) VALUES"
                       "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",row)
    mysql.connection.commit()
    cursor.close()
    return "Database created"
@mysql_test.route('/retrieve')
def retrieve():

    cursor = mysql.connection.cursor()
    target_image = "DSC00200.JPG"

    cursor.execute("SELECT * FROM test WHERE picNUM = %s", [target_image])
    #cursor.execute("SELECT * FROM test")
    data = jsonify(cursor.fetchall())

    return data

if __name__ == '__main__':
    mysql_test.run()


