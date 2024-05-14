from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/cifar-10'

    # Liên kết ứng dụng với cơ sở dữ liệu
    db.init_app(app)

    return app
