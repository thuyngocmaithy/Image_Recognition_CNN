# Plot ad hoc CIFAR10 instances
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import numpy as np
from models import Image  as ImageModel
from create_app import create_app, db

app = create_app()

def label_to_index(label):
    labels = {
        'airplane': 0,
        'automobile': 1,
        'bird': 2,
        'cat': 3,
        'deer': 4,
        'dog': 5,
        'frog': 6,
        'horse': 7,
        'ship': 8,
        'truck': 9
    }
    return labels[label]
# load data
# Mỗi hình ảnh được biểu diễn dưới dạng ma trận ba chiều, 
# với các kích thước là đỏ, lục, lam, chiều rộng và chiều cao
# tải tập dữ liệu lên 2 mảng train và test 
(X_train, y_train), (X_test, y_test) = cifar10.load_data() 
# Tạo lưới hình ảnh 3x3
for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    plt.imshow(X_train[i])
# hiển thị hình ảnh
plt.show()


# CNN model for the CIFAR-10 Dataset
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import MaxNorm
from keras.optimizers import SGD
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.utils import to_categorical


# load data
# Các giá trị pixel nằm trong khoảng từ 0 đến 255 
# cho mỗi kênh màu đỏ, xanh lục và xanh lam.
# Load CIFAR-10 dataset
(X_cifar_train, y_cifar_train), (X_cifar_test, y_cifar_test) = cifar10.load_data()
# normalize inputs from 0-255 to 0.0-1.0
# dữ liệu được tải dưới dạng int
# => phải chuyển nó sang float để thực hiện phép chia.
# Preprocess CIFAR-10 data
X_cifar_train = X_cifar_train.astype('float32') / 255.0
X_cifar_test = X_cifar_test.astype('float32') / 255.0
# chuyển đổi về dạng one-hot encoding
y_cifar_train = to_categorical(y_cifar_train)
y_cifar_test = to_categorical(y_cifar_test)

with app.app_context():
    # Thực hiện lấy các hình ảnh trong database
    images = ImageModel.query.all()

# Xử lý images and labels from the database
X_db = []
y_db = []
for image in images:
    print(image)
    # Load tệp hình ảnh từ hệ thống tệp" 
    image_path = f'uploads/{image.filename}'
    image_data = PILImage.open(image_path)
    # Thay đổi kích thước hình ảnh cho phù hợp với đầu vào input
    image_data = image_data.resize((32, 32))
    # Convert image to numpy array and normalize
    image_array = np.array(image_data) / 255.0
    X_db.append(image_array)
    # Convert label name to index
    image_id = image.id

    # Retrieve the Image instance within a session context
    with app.app_context():
        image = ImageModel.query.get(image_id)

        label_index = label_to_index(image.label.name)
    print(label_index)
    
    y_db.append(label_index)

# Convert to numpy arrays
X_db = np.array(X_db)
y_db = np.array(y_db)

# Combine CIFAR-10 data and database data
y_db_one_hot = to_categorical(y_db, num_classes=10)

# Combine CIFAR-10 data and database data
X_combined = np.concatenate((X_cifar_train, X_cifar_test, X_db), axis=0)
y_combined = np.concatenate((y_cifar_train, y_cifar_test, y_db_one_hot), axis=0)



# Tạo model
model = Sequential()
# Lớp tích chập với 32 bộ lọc có kích thước 3x3, 
# sử dụng hàm kích hoạt ReLU 
# và thêm các đệm sao cho đầu ra có kích thước bằng với đầu vào.
# ####################BEGIN BỘ BA CONVULATION LAYER + NONLINEAR LAYER + POOLING LAYER 1####################
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
# Lớp Dropout để ngẫu nhiên "bỏ qua" 20% các nơ-ron trong quá trình huấn luyện
# nhằm tránh tình trạng quá khớp (overfitting).
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# Lớp gộp Max pooling với kích thước cửa sổ mặc định là 2x2, giảm kích thước đầu vào
model.add(MaxPooling2D())
# #####################END BỘ BA CONVULATION LAYER + NONLINEAR LAYER + POOLING LAYER 1#####################
# ####################BEGIN BỘ BA CONVULATION LAYER + NONLINEAR LAYER + POOLING LAYER 2####################
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D())
# #####################END BỘ BA CONVULATION LAYER + NONLINEAR LAYER + POOLING LAYER 2#####################
# ####################BEGIN BỘ BA CONVULATION LAYER + NONLINEAR LAYER + POOLING LAYER 3####################
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D())
# #####################END BỘ BA CONVULATION LAYER + NONLINEAR LAYER + POOLING LAYER 3#####################
# #####################PHẲNG HÓA#####################
model.add(Flatten())
model.add(Dropout(0.2))
# #####################FULLY CONNECTED#####################
# Lớp fully connected với 1024 nơ-ron và hàm kích hoạt ReLU. 
# Thêm ràng buộc kernel với MaxNorm để hạn chế độ lớn của các trọng số đầu ra tại mỗi bước cập nhật.
model.add(Dense(1024, activation='relu', kernel_constraint=MaxNorm(3)))
model.add(Dropout(0.2))
# Lớp fully connected khác với 512 nơ-ron và hàm kích hoạt ReLU, cũng có ràng buộc kernel với MaxNorm.
model.add(Dense(512, activation='relu', kernel_constraint=MaxNorm(3)))
model.add(Dropout(0.2))
# Lớp fully connected cuối cùng với số nơ-ron bằng số lớp đầu ra (num_classes), 
# sử dụng hàm kích hoạt softmax để ánh xạ các đầu vào thành xác suất cho mỗi lớp đầu ra.
model.add(Dense(10, activation='softmax'))
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(learning_rate=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()
# Huấn luyện mô hình
model.fit(X_combined, y_combined, epochs=epochs, batch_size=32, validation_split=0.2)

# Lưu mô hình
model.save("model1_cifar_10epoch.h5")
# Final evaluation of the model
# Convert y_test to one-hot encoded format
y_test_one_hot = to_categorical(y_test, num_classes=num_classes)

# Evaluate the model with one-hot encoded y_test
scores = model.evaluate(X_test, y_test_one_hot, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))