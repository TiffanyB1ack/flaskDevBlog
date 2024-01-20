from flask import render_template,request
from app.main import bp
from flask_login import login_required, current_user
from app.models.post import Post
from app.extensions import db

import config

@bp.route('/')
def index():
    posts = Post.query.all()
    page = request.args.get('page', 1, type=int)
    pagination = Post.query.order_by(Post.id).paginate(page=page, per_page=config.POSTS_PER_PAGE)
    return render_template("index.html",posts=posts,pagination=pagination)

@bp.route('/profile')
@login_required
def profile():
    return render_template('auth/profile.html', nickname=current_user.nickname, image=current_user.image, email=current_user.email)