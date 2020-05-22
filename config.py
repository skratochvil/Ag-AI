"""@package config
This module handles configuration fro the web application.
"""
import os

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'