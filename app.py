import os
import time
import uuid
from flask import Flask, redirect, request, render_template, jsonify, url_for
from create_app import create_app
from models import Label, Image as ImageModel
from predict import predict_image_class
from werkzeug.utils import secure_filename  
from PIL import Image
import numpy as np
from models import db
import subprocess

app = create_app()



# Đường dẫn tuyệt đối đến thư mục lưu trữ tệp tin đã tải lên
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add-data')
def add_data():
    return render_template('add-data.html')

@app.route('/predict_image_class', methods=['POST'])
def predict_image():
    data = request.get_json()
    image_path = data.get('image_path')    
    prediction = predict_image_class(image_path)    
    return jsonify({'prediction': prediction})


@app.route('/upload', methods=['POST'])
def upload_file():
    results = {
        0: ['airplane', 'Máy bay'],
        1: ['automobile', 'Ô tô'],
        2: ['bird', 'Chim'],
        3: ['cat', 'Mèo'],
        4: ['deer', 'Nai'],
        5: ['dog', 'Chó'],
        6: ['frog', 'Ếch'],
        7: ['horse', 'Ngựa'],
        8: ['ship', 'Tàu thủy'],
        9: ['truck', 'Xe tải']
    }
    if request.method == 'POST':
        file = request.files.get('file')
        label_name = request.form.get('label')
        
        for key, values in results.items():
            if(values[0] == label_name):
                label_name_tieng_viet = values[1]
                
        if file is None or label_name_tieng_viet is None:
            return jsonify({'result': 'failure', 'message': 'Dữ liệu không hợp lệ'})

        # Kiểm tra xem nhãn đã tồn tại trong cơ sở dữ liệu chưa
        label = Label.query.filter_by(name=label_name).first()
        if label is None:
            label = Label(name=label_name, vietnamese_name= label_name_tieng_viet)
            db.session.add(label)
            db.session.commit()

        # Lưu tên tệp và nhãn vào cơ sở dữ liệu
        filename = secure_filename(file.filename)
        image = ImageModel(filename=filename, label=label)
        db.session.add(image)
        db.session.commit()

        # Lưu file vào thư mục uploads
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        return jsonify({'result': 'success'})

    return jsonify({'result': 'failure', 'message': 'Invalid request method'})


@app.route('/train', methods=['POST'])
def train():
    try:
        # Execute the train_model.py file as a subprocess
        subprocess.run(['python', 'train_model.py'])
        return jsonify({'status': 'success', 'message': 'Training completed successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

def generate_filename():
    # Tạo tên file động bằng cách sử dụng timestamp và UUID
    timestamp = int(time.time())  # Sử dụng timestamp làm phần đầu của tên file
    unique_id = str(uuid.uuid4())  # Sử dụng UUID động để đảm bảo tính duy nhất
    filename = f"{timestamp}_{unique_id}.png"
    return filename

    
if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)
