"""@package __init__
This module is responsible for linking the application together.
"""
from flask import Flask
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

from app import web