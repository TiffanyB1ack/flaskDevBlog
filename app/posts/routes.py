from datetime import datetime as dt

from flask import render_template, request, redirect, url_for,flash
from flask_login import login_required, current_user
from app.models.user import User
from app.extensions import db
from app.models.post import Post
from app.posts import bp


@bp.route('/', methods=('GET', 'POST'))
@login_required
def index():
    posts = Post.query.all()
    if request.method == "POST":
        new_post = Post(content=request.form['content'],
                        author=current_user.nickname,
                        title=request.form['title'],
                        date_post=dt.now(),
                        author_image=User.query.filter_by(id=current_user.id).first().image
                        )
        db.session.add(new_post)
        db.session.commit()
        flash('Ваш пост был создан. Переидите на главную, чтобы его увидеть')
        return redirect(url_for('posts.index'))

    return render_template("posts/index.html", posts=posts)

