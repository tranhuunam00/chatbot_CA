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

# 2. Load tất cả tài liệu từ thư mục "data"
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

# 6. Hàm gọi Gemini API để sinh văn bản


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

# 7. Hàm tạo prompt từ câu hỏi và nội dung tài liệu


def create_prompt(query: str, docs: list) -> str:
    data = "\n\n".join([doc.page_content for doc in docs])
    return f"""
Bạn là một trợ lý ảo được huấn luyện chuyên sâu về **Y học giấc ngủ** và các kiến thức giảng dạy trong chương trình đào tạo 6 tháng của **Hội Y học Giấc ngủ Việt Nam**.

Dưới đây là các thông tin trích xuất từ tài liệu chính thức:

{data}

Hãy sử dụng các thông tin trên để trả lời câu hỏi sau bằng tiếng Việt, rõ ràng, chính xác và có căn cứ khoa học:
"{query}"

⚠️ Yêu cầu:
- Trả lời đúng nội dung chuyên ngành y học giấc ngủ, đặc biệt các chủ đề như: sinh lý học giấc ngủ, các rối loạn giấc ngủ, kỹ thuật chẩn đoán, điều trị bằng CPAP, kỹ năng thực hành lâm sàng, đánh giá đầu ra, cấu trúc khóa học, v.v.
- Nếu không đủ thông tin để trả lời, hãy nêu rõ điều đó một cách lịch sự và trung thực.
- Trả lời có cấu trúc, rõ ràng, dễ hiểu, đúng với chương trình đào tạo và thuật ngữ chuyên ngành.
"""


# 8. Demo truy vấn thử
query = "Tổng số tiết thực hành trong chương trình là bao nhiêu?"
docs = db.similarity_search(query, k=3)
prompt = create_prompt(query, docs)
response = generate_from_prompt(prompt)

print("\n🤖 Câu trả lời từ Gemini:")
print(response)
