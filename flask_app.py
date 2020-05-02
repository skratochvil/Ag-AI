"""@package flask_app
This module handles the web apps and acts as the main driver for running this web application.

"""
from app import app

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug ='True')    