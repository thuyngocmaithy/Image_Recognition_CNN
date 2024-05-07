# Plot ad hoc CIFAR10 instances
from keras.datasets import cifar10
import matplotlib.pyplot as plt
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
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# normalize inputs from 0-255 to 0.0-1.0
# dữ liệu được tải dưới dạng int
# => phải chuyển nó sang float để thực hiện phép chia.
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# chuyển đổi về dạng one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

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
model.add(Dense(num_classes, activation='softmax'))
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(learning_rate=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()
# Huấn luyện mô hình
model.fit(X_train, y_train, validation_data=(X_test, y_test), 
          epochs=epochs, batch_size=32)
model.save("model1_cifar_10epoch.h5")
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))