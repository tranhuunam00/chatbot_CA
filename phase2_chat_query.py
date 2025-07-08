import os
import requests
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 1. Load biến môi trường
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise EnvironmentError("❌ Thiếu GOOGLE_API_KEY trong file .env")

# 2. Load FAISS Vector DB đã lưu từ Phase 1
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = FAISS.load_local("faiss_yhgn_db", embedding,
                      allow_dangerous_deserialization=True)

print("✅ FAISS Vector Store đã được tải")

# 3. Hàm gọi Gemini API nếu cần diễn giải thêm


def generate_from_prompt(prompt: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print("🌐 Lỗi kết nối Gemini API:", e)
        return "❌ Không thể lấy câu trả lời từ Gemini."

# 4. Hàm tạo prompt tổng hợp từ tài liệu và câu hỏi


def create_prompt(query: str, docs: list) -> str:
    context = "\n\n".join([doc.page_content for doc in docs])
    return f"""
Bạn là trợ lý chuyên ngành Y học Giấc ngủ, dựa vào tài liệu dưới đây để trả lời câu hỏi của học viên trong chương trình đào tạo 6 tháng do Hội Y học Giấc ngủ Việt Nam tổ chức.

Dữ liệu tài liệu:
{context}

Câu hỏi: "{query}"

Hãy trả lời rõ ràng, đúng chuyên môn, mạch lạc và súc tích bằng tiếng Việt.
"""

# 5. Hàm xử lý truy vấn người dùng


def query_system(user_input: str):
    docs = db.similarity_search(user_input, k=3)

    # Kiểm tra xem có đoạn nào "sát nghĩa" không
    matched_directly = [
        doc for doc in docs if user_input.lower() in doc.page_content.lower()
    ]

    if matched_directly:
        print("✅ Trả lời trực tiếp từ tài liệu:")
        print(matched_directly[0].page_content)
    else:
        print("🤖 Không thấy đoạn sát nghĩa. Đang hỏi Gemini...")
        prompt = create_prompt(user_input, docs)
        response = generate_from_prompt(prompt)
        print(response)


# 6. Giao diện dòng lệnh đơn giản
if __name__ == "__main__":
    print("🤖 Trợ lý Y học Giấc ngủ đã sẵn sàng. Nhập câu hỏi hoặc gõ 'exit' để thoát.")
    while True:
        user_input = input("\n❓ Câu hỏi: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("👋 Tạm biệt!")
            break
        query_system(user_input)
