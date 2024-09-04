from app.extensions import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)


class UserKeystrokes(db.Model):
    # Table for storing keystroke data
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False,unique=True)
    keystroke_path = db.Column(db.String(120), nullable=False)
    features_path = db.Column(db.String(120), nullable=False)

class TrainedModels(db.Model):
    # Table for storing trained model paths
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)
    model_path = db.Column(db.String(120), nullable=False)