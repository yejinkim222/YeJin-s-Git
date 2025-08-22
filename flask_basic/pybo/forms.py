from random import choices
from wsgiref.validate import validator

from flask_wtf import FlaskForm
from wtforms.fields.simple import StringField, TextAreaField, PasswordField
from wtforms.validators import DataRequired, Length, EqualTo, Email, NumberRange, InputRequired, Optional
from wtforms import IntegerField, RadioField, SelectField, HiddenField


class InputDataForm(FlaskForm):
    # 사용자 입력
    age = IntegerField('Age', validators=[DataRequired('나이는 필수 입력 항목입니다. 65~120의 값으로 입력해주세요.'), NumberRange(min=65, max=120)])
    gender = RadioField('Gender', choices=[('0','남성'),('1','여성')], coerce=int, validators=[InputRequired('성별은 필수 입력 항목입니다.')])
    edu_level = SelectField('Edu_level', choices=[('0','무학'),('1','초졸'),('2','중졸'),('3','고졸'),('4','대졸 이상')], coerce=int, validators=[InputRequired('교육 수준은 필수 입력 항목입니다.')])
    has_db = RadioField('Has_db',choices=[('0','없음'),('1','있음')], coerce=int, validators=[InputRequired('당뇨병 여부는 필수 입력 항목입니다.')])
    has_hibpe = RadioField('Has_hibpe',choices=[('0','없음'),('1','있음')], coerce=int, validators=[InputRequired('고혈압 여부는 필수 입력 항목입니다.')])
    has_mci = HiddenField('Has_mci', validators=[Optional()])
    base_yrs = IntegerField('Base_yrs', validators=[DataRequired('예측 기준 연도는 필수 입력 항목입니다. 1~10의 값으로 입력해주세요.'), NumberRange(min=1, max=10)])

class ModelDataForm(FlaskForm):
    id = StringField('Id', validators = [DataRequired()])
    input_id = StringField('Input_id', validators = [DataRequired()])
    age = StringField('Age', validators = [DataRequired()])
    gender = StringField('Gender', validators = [DataRequired()])
    edu_yrs = StringField('Edu_yrs', validators = [DataRequired()])
    has_db = StringField('Has_db', validators = [DataRequired()])
    AD_MCI_status = StringField('AD_MCI_status', validators = [DataRequired()])
    has_hibpe = StringField('Has_hibpe', validators = [DataRequired()])
    edu_level = StringField('Edu_level', validators = [DataRequired()])
    db_onset_after = StringField('Db_onset_after', validators = [DataRequired()])
    mci_onset_after = StringField('Mci_onset_after', validators = [DataRequired()])
    hibpe_onset_after = StringField('Hibpe_onset_after', validators = [DataRequired()])
    age_group5 = StringField('Age_group5', validators = [DataRequired()])
    risk_factor_sum = StringField('Risk_factor_sum', validators = [DataRequired()])
    edu_is_low = StringField('Edu_is_low', validators = [DataRequired()])
    risk_weighted_age = StringField('Risk_weighted_age', validators = [DataRequired()])
    age_gender_interact = StringField('Age_gender_interact', validators = [DataRequired()])
    hibpe_onset_after_missing = StringField('Hibpe_onset_after_missing', validators = [DataRequired()])
    has_hibpe_missing = StringField('Has_hibpe_missing', validators = [DataRequired()])
    mci_onset_after_missing = StringField('Mci_onset_after_missing', validators = [DataRequired()])
    edu_yrs_missing = StringField('Edu_yrs_missing', validators = [DataRequired()])
    db_onset_after_missing = StringField('Db_onset_after_missing', validators = [DataRequired()])
    cognitive_decline_flag = StringField('Cognitive_decline_flag', validators = [DataRequired()])
    age_x_edu = StringField('Age_x_edu', validators = [DataRequired()])
    hibpe_onset_delay_ratio = StringField('Hibpe_onset_delay_ratio', validators = [DataRequired()])
    age_edu_ratio = StringField('Age_edu_ratio', validators = [DataRequired()])

class UserCreateForm(FlaskForm):
    username = StringField('사용자이름', validators=[DataRequired(), Length(min=3, max=25)])
    password1 = PasswordField('비밀번호', validators=[
        DataRequired(), EqualTo('password2', '비밀번호가 일치하지 않습니다')])
    password2 = PasswordField('비밀번호확인', validators=[DataRequired()])
    email = StringField('이메일', validators=[DataRequired(), Email()])

class UserLoginForm(FlaskForm):
    username = StringField('사용자이름', validators=[DataRequired(), Length(min=3,max=25)])
    password = PasswordField('비밀번호', validators=[DataRequired()])

