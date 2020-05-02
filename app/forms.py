"""@package forms
This module contains form objects that are responsible for the dynamic parts of the webpages.
These forms also can handle user input.
"""

from flask_wtf import FlaskForm
from wtforms import RadioField, SubmitField
from wtforms.validators import DataRequired

class LabelForm(FlaskForm):
    """
    This class contains a two option radio button and a submit button.
    This class is to be used to allow the user to label a picture as healthy and unhealthy.
    
    """
    choice = RadioField(u'Label', choices=[('H', u'Healthy'), ('B', u'Unhealthy')], validators = [DataRequired()])
    submit = SubmitField('Add Label')
