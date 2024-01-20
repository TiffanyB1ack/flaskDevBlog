from flask import render_template,url_for
from app.about import bp



@bp.route('/')
def index():

    return render_template("about/index.html")


