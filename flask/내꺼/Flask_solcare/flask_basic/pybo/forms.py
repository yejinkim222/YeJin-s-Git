from wsgiref.validate import validator

from flask_wtf import FlaskForm
from wtforms.fields.simple import StringField, TextAreaField, PasswordField
from wtforms.validators import DataRequired, Length, EqualTo, Email

# 내꺼에 맞게 고치기
class InputDataForm(FlaskForm):
    age = StringField('Age', validators=[DataRequired('나이는 필수 입력 항목입니다.')])
    gender = StringField('Gender', validator=[DataRequired('성별은 필수 입력 항목입니다.')])
    edu_yrs = StringField('Edu_yrs', validator=[DataRequired('교육 연도는 필수 입력 항목입니다.')])
    has_db = StringField('Has_Db', validator=[DataRequired('당뇨병 여부는 필수 입력 항목입니다.')])
    has_hibpe = StringField('Has_Hibpe', validator=[DataRequired('고혈압 여부는 필수 입력 항목입니다.')])
    has_mci = StringField('Has_Mci', validator=[DataRequired('경도 인지 장애 여부는 필수 입력 항목입니다.')])
    base_yrs = StringField('Base_yrs', validator=[DataRequired('예측 기준 연도는 필수 입력 항목입니다.')])

class AnswerForm(FlaskForm):


class UserCreateForm(FlaskForm):
    username = StringField('사용자이름', validators=[DataRequired(), Length(min=3, max=25)])
    password1 = PasswordField('비밀번호', validators=[
        DataRequired(), EqualTo('password2', '비밀번호가 일치하지 않습니다')])
    password2 = PasswordField('비밀번호확인', validators=[DataRequired()])
    email = StringField('이메일', validators=[DataRequired(), Email()])

class UserLoginForm(FlaskForm):
    username = StringField('사용자이름', validators=[DataRequired(), Length(min=3,max=25)])
    password = PasswordField('비밀번호', validators=[DataRequired()])

