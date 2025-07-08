# 🧠 Sleep ChatBot

Chatbot AI trả lời các câu hỏi dựa trên văn bản giấc ngủ sử dụng LangChain + OpenAI.

## ✅ Cài đặt

```bash
pip install -r requirements.txt
```

python3.13 -m pip install -r requirements.txt

deploy

✅ Cách 1: Cài Python 3.13 từ source (ổn định và phổ biến)
🔹 Bước 1: Cài các gói phụ thuộc
bash
Sao chép
Chỉnh sửa
sudo apt update
sudo apt install -y build-essential libssl-dev zlib1g-dev \
 libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
 libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev \
 tk-dev uuid-dev wget
🔹 Bước 2: Tải source Python 3.13 (hoặc RC)
bash
Sao chép
Chỉnh sửa
cd /usr/src
sudo wget https://www.python.org/ftp/python/3.13.0/Python-3.13.0b2.tgz
sudo tar xzf Python-3.13.0b2.tgz
cd Python-3.13.0b2
📌 Lưu ý: Ở thời điểm hiện tại (tháng 7/2025), bản mới nhất có thể là bản beta hoặc RC, ví dụ: 3.13.0b2
Kiểm tra bản mới nhất tại: https://www.python.org/ftp/python/

🔹 Bước 3: Biên dịch và cài đặt
bash
Sao chép
Chỉnh sửa
sudo ./configure --enable-optimizations
sudo make -j$(nproc)
sudo make altinstall
Dùng make altinstall để không ghi đè python3 mặc định của Ubuntu

🔹 Bước 4: Kiểm tra
bash
Sao chép
Chỉnh sửa
python3.13 --version

python3.13 -m ensurepip --upgrade
