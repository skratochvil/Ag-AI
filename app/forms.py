from flask_wtf import FlaskForm, Required
from wtforms import RadioField, SubmitField

class LabelForm(FlaskForm):
    choice = RadioField(u'Label', choices=[('1', u'Healthy'), ('2', u'UnHealthy')], validators = [Required()])
    submit = SubmitField('Add Label')