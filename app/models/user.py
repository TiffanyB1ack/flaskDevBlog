from app.extensions import db
from flask_login import UserMixin

class User(UserMixin,db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nickname= db.Column(db.String(150))
    email = db.Column(db.String(150),unique=True)
    password=db.Column(db.String(500))
    image=db.Column(db.String(300),default='media/profile_foto/default.png')
