from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from create_app import create_app, db

app = create_app()
class Label(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    vietnamese_name = db.Column(db.String(100), nullable=True)

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), unique=True, nullable=False)
    label_id = db.Column(db.Integer, db.ForeignKey('label.id'), nullable=False)
    label = db.relationship('Label', backref=db.backref('images', lazy=True))

# Tạo cơ sở dữ liệu (chỉ cần chạy một lần)
with app.app_context():
    db.create_all()