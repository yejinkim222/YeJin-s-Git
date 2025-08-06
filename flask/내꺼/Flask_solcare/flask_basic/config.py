import os

BASE_DIR = os.path.dirname(__file__)
print("BASE_DIR:",BASE_DIR)

SQLALCHEMY_DATABASE_URI = 'sqlite:///{}'.format(os.path.join(BASE_DIR,'solcare.db'))

print("SQLALCHEMY_DATABASE_URI:",SQLALCHEMY_DATABASE_URI)
SQLALCHEMY_TRACK_MODIFICATIONS = False

# 시크릿 키는 암호 만들려면 해야하는데 아직 안함
# SECRET_KEY="dev"
