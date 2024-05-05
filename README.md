
# Giới thiệu Website ReT - Nhận dạng hình ảnh bằng AI:

## Giới thiệu
Website nhận dạng ảnh bằng AI sử dụng mô hình CNN với sự hỗ trợ của thư viện Keras giúp nhận biết được các lớp ảnh "airplane", "automobile", "bird", "cat", "deer", "frog", "horse", "ship" và "truck".  
Xem thêm: https://phuongthuy2512.github.io/Image_Recognition_WEB/

## Hướng dẫn cài đặt
### Bước 1: Tải xuống mã nguồn của ReT từ Github:
- Truy cập vào trang Github của dự án tại đây: https://github.com/thuyngocmaithy/Image_Recognition_CNN
- Chọn Download ZIP
### Bước 2: Cài đặt Python (Nếu chưa có Python):
- Windows:
Truy cập đường dẫn sau: https://www.python.org/downloads/  
Tải file .exe và tiến hành cài đặt trên máy tính.  
- Linux:  
Mở Terminal và nhập lệnh:
```bash
  sudo apt install python3
```
Tiếp theo tiến hành cài đặt pip:
```bash
  sudo apt install -y python3-pip
```
Nâng cấp pip và setuptools mới nhất:
```bash
  pip3 install --upgrade pip
  pip3 install --upgrade setuptool
```
### Bước 3: Cài đặt các thư viện cần thiết:
Bạn có thể cài đặt các thư viện cần thiết bằng cách sử dụng file requirements.txt đi kèm với mã nguồn của ReT. Mở Command Prompt hoặc Terminal và điều hướng đến thư mục chứa mã nguồn của ứng dụng, sau đó chạy lệnh sau để cài đặt các thư viện cần thiết:  
Mở Terminal và nhập lệnh:
```bash
  pip install -r requirements.txt
```
### Bước 4: Chạy ứng dụng:
Sau khi cài đặt Python và các thư viện cần thiết, bạn có thể chạy website nhận dạng hình ảnh CNN - ReT. Mở Command Prompt hoặc Terminal và điều hướng đến thư mục chứa mã nguồn của ứng dụng.  
Chạy lệnh sau để khởi chạy ứng dụng:
```bash
  python main.py
```
Hoặc
```bash
  python3 main.py
```