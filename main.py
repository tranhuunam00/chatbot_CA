import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load biến môi trường
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise EnvironmentError("❌ Thiếu GOOGLE_API_KEY trong file .env")

# Load FAISS VectorStore
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = FAISS.load_local("faiss_yhgn_db", embedding,
                      allow_dangerous_deserialization=True)
print("✅ FAISS Vector Store đã được tải")

# Tạo FastAPI app
app = FastAPI(title="YHGN Assistant API")

# Schema cho request body


class QuestionRequest(BaseModel):
    query: str

# Hàm gọi Gemini nếu cần


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
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print("🌐 Lỗi kết nối Gemini API:", e)
        return "❌ Không thể lấy câu trả lời từ Gemini."

# Hàm tạo prompt


def create_prompt(query: str, docs: list) -> str:
    context = "\n\n".join([doc.page_content for doc in docs])
    return f"""
Bạn là trợ lý chuyên ngành Y học Giấc ngủ, dựa vào tài liệu dưới đây để trả lời câu hỏi của học viên trong chương trình đào tạo 6 tháng do Hội Y học Giấc ngủ Việt Nam tổ chức.

Dữ liệu tài liệu:
{context}

Câu hỏi: "{query}"

Hãy trả lời rõ ràng, đúng chuyên môn, mạch lạc và súc tích bằng tiếng Việt.
"""

# Endpoint chính


@app.post("/ask")
def ask_question(data: QuestionRequest):
    query = data.query.strip()
    if not query:
        raise HTTPException(
            status_code=400, detail="Query không được để trống.")

    docs = db.similarity_search(query, k=3)
    matched_directly = [
        doc for doc in docs if query.lower() in doc.page_content.lower()]

    if matched_directly:
        return {
            "answer_source": "direct_match",
            "answer": matched_directly[0].page_content
        }
    else:
        prompt = create_prompt(query, docs)
        response = generate_from_prompt(prompt)
        return {
            "answer_source": "gemini",
            "answer": response
        }
