# main.py

from predict import predict_image_class

# Provide the path to the image you want to classify
image_path = "horse.jpg"

# Make prediction
prediction = predict_image_class(image_path)
print("Predicted Class:", prediction)
