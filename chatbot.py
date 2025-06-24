import os
import requests
from dotenv import load_dotenv
import glob

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 1. Load biến môi trường
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise EnvironmentError("❌ Thiếu GOOGLE_API_KEY trong file .env")

# 2. Load tất cả tài liệu từ folder "data"
all_documents = []
file_paths = glob.glob("data/*.txt")
for path in file_paths:
    loader = TextLoader(path, encoding="utf-8")
    all_documents.extend(loader.load())
print(f"✅ Đã load {len(file_paths)} file văn bản")

# 3. Chia nhỏ văn bản để index
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = splitter.split_documents(all_documents)
print(f"✅ Đã chia thành {len(chunks)} đoạn")

# 4. Tạo embedding bằng Gemini
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 5. Tạo FAISS vector store
db = FAISS.from_documents(chunks, embedding)
print("✅ Đã tạo FAISS vector store thành công")

# 6. Hàm gọi Gemini API sinh văn bản (dùng gemini-2.0-flash)


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
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        return text
    except Exception as e:
        print("🌐 Lỗi kết nối Gemini API:", e)
        return "❌ Lỗi không truy xuất được câu trả lời từ Gemini."

# 7. Hàm tạo prompt từ câu hỏi và dữ liệu truy vấn


def create_prompt(query: str, docs: list) -> str:
    data = "\n\n".join([doc.page_content for doc in docs])
    return f"""
Bạn là một trợ lý ảo được huấn luyện chuyên sâu về **thủ tục hành chính và các quy định pháp luật trong lĩnh vực Công an** Việt Nam.
Dưới đây là các thông tin đã được trích xuất từ tài liệu chính thức (văn bản pháp luật, hướng dẫn từ Bộ Công an):

{data}

Hãy sử dụng các thông tin trên để trả lời cho câu hỏi sau bằng tiếng Việt, rõ ràng, chính xác và có căn cứ pháp lý:
"{query}"

⚠️ Yêu cầu:
- Ưu tiên các nội dung liên quan đến thủ tục hành chính (hộ khẩu, CCCD, cư trú, xử phạt hành chính, xuất nhập cảnh, đăng ký phương tiện, v.v.).
- Nếu câu hỏi liên quan đến hình sự, quy định pháp luật, hành vi vi phạm,... vẫn có thể trả lời nếu nằm trong phạm vi các tài liệu đã cung cấp.
- Nếu không đủ thông tin để trả lời, hãy nói rõ điều đó một cách lịch sự và trung thực.
- Trả lời có cấu trúc, chính xác, đúng quy định, có thể nêu các bước, hồ sơ, mức phạt hoặc điều khoản pháp luật tương ứng.
"""


# 8. Demo truy vấn
query = "Thông tin trên giấy tờ xuất nhập cảnh bao gồm"
docs = db.similarity_search(query, k=3)
prompt = create_prompt(query, docs)
response = generate_from_prompt(prompt)

print("\n🤖 Câu trả lời từ Gemini:")
print(response)
