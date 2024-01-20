from app.extensions import db
import datetime

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(150))
    content = db.Column(db.Text)
    author = db.Column(db.String(150))
    date_post = db.Column(db.Date)
