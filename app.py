from flask import Flask, request, render_template, jsonify
from predict import predict_image_class

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_image_class', methods=['POST'])
def predict_image():
    data = request.get_json()
    image_path = data.get('image_path')    
    prediction = predict_image_class(image_path)    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
