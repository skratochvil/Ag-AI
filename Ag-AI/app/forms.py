from flask_wtf import FlaskForm
from wtforms import RadioField, SubmitField
from wtforms.validators import DataRequired

class LabelForm(FlaskForm):
    choice = RadioField(u'Label', choices=[('H', u'Healthy'), ('B', u'UnHealthy')], validators = [DataRequired()])
    submit = SubmitField('Add Label')