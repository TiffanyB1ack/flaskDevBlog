from flask import render_template, redirect, url_for, request, flash, session
from flask_login import login_user, login_required, logout_user
from os.path import join
import config
from app.auth import bp
from app.extensions import db
from app.models.user import User
from werkzeug.utils import secure_filename
import os

@bp.route('/login')
def login():
    return render_template('auth/login.html',)


@bp.route('/login', methods=['POST'])
def login_post():
    email = request.form.get('email')
    password = request.form.get('password')
    remember = True if request.form.get('remember') else False

    user = User.query.filter_by(email=email).first()
    if not user or user.password!=password:
        flash('Вы ввели неверный логин и/или пароль')
        return redirect(url_for('main.index'))

    login_user(user, remember=remember)

    return redirect(url_for('main.profile'))


@bp.route('/signup')
def signup():
    return render_template('auth/singup.html')


@bp.route('/signup', methods=['POST'])
def signup_post():
    email = request.form.get('email')
    name = request.form.get('nickname')
    password = request.form.get('password')
    password2 = request.form.get('password2')
    image = 'default.png'
    user = User.query.filter_by(
        email=email).first()

    if user:
        flash('Email address already exists')
        return redirect(url_for('auth.signup'))

    if password == password2:

        new_user = User(email=email, nickname=name, password=password, image=image)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('auth.login'))
    else:
        flash('Пароли должны быть разные')
        return redirect(url_for('auth.signup'))

@bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.index'))






