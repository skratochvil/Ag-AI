from flask_wtf import FlaskForm
from wtforms import RadioField, SubmitField
from wtforms.validators import DataRequired

class LabelForm(FlaskForm):
    choice = RadioField(u'Label', choices=[('1', u'Healthy'), ('2', u'UnHealthy')], validators = [DataRequired()])
    submit = SubmitField('Add Label')