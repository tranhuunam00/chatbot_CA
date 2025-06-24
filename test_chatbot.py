import os
import requests
from dotenv import load_dotenv

# 1. Load API key từ .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise EnvironmentError("❌ Thiếu GOOGLE_API_KEY trong file .env")

# 2. Thiết lập endpoint và headers
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
headers = {"Content-Type": "application/json"}

# 3. Prompt để kiểm tra
prompt = "Giải thích ngắn gọn cách hoạt động của trí tuệ nhân tạo (AI)"

data = {
    "contents": [
        {
            "parts": [
                {"text": prompt}
            ]
        }
    ]
}

# 4. Gửi POST request
try:
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    content = result["candidates"][0]["content"]["parts"][0]["text"]
    print("✅ Kết quả từ Gemini:\n")
    print(content)
except Exception as e:
    print("❌ Đã xảy ra lỗi:", e)
    print(response.text)
