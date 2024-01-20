class Config:
    SQLALCHEMY_DATABASE_URI='mysql://root:password@localhost/FlaskDevBlog'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    LANGUAGES = ['en', 'ru']

    # pagination
POSTS_PER_PAGE = 3
UPLOAD_FOLDER='static/media/profile_foto/'