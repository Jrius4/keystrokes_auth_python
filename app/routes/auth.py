from flask import Blueprint, render_template

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login')
def login():
    return render_template('login.html')


@auth_bp.route('/loginCnn')
def loginCnn():
    return render_template('login_cnn.html')



@auth_bp.route('/loginMlp')
def loginMlp():
    return render_template('login_mlp.html')


@auth_bp.route('/loginRf')
def loginRf():
    return render_template('login_rf.html')

@auth_bp.route('/loginGan')
def loginGan():
    return render_template('login_gan.html')

@auth_bp.route('/loginLstm')
def loginLstm():
    return render_template('login_lstm.html')
