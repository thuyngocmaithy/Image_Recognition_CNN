# predict.py

from keras.models import load_model
from PIL import Image
import numpy as np
import base64
import io

# Load the saved model
model = load_model("model1_cifar_10epoch.h5")

# Define class labels
results = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

def predict_image_class(image_path):
    # Load and preprocess the image
    # Giải mã dữ liệu base64 và chuyển đổi thành hình ảnh
    im = Image.open(io.BytesIO(base64.b64decode(image_path.split(',')[1])))
    im = im.resize((32, 32))
    im = np.expand_dims(im, axis=0)
    im = np.array(im) / 255.0  # Normalize pixel values

    # Make prediction
    pred = model.predict(im)
    predicted_class = np.argmax(pred, axis=1)[0]
    return results[predicted_class]
