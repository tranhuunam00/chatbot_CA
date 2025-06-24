import os
import requests
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 1. Load biến môi trường
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise EnvironmentError("❌ Thiếu GOOGLE_API_KEY trong file .env")

# 2. Load tài liệu từ file
file_path = "data/raw.txt"
loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()
print("✅ Đã load tài liệu")

# 3. Chia nhỏ văn bản để index
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
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
    Bạn là một chuyên gia trí tuệ nhân tạo chuyên về **pháp luật và thủ tục trong lĩnh vực Công an**.
    Dưới đây là các thông tin đã được trích xuất từ tài liệu pháp luật và hướng dẫn chính thức:

    {data}

    Dựa vào các thông tin trên, hãy trả lời câu hỏi sau bằng tiếng Việt:
    "{query}"

    ⚠️ Lưu ý:
    - Chỉ trả lời nếu câu hỏi liên quan đến lĩnh vực Công an, bao gồm: thủ tục hành chính, pháp luật hình sự, an ninh trật tự, xử lý vi phạm, v.v.
    - Nếu không liên quan, hãy trả lời: "Xin lỗi, tôi chỉ hỗ trợ các câu hỏi liên quan đến lĩnh vực Công an."
    - Trả lời chính xác, rõ ràng, có dẫn chiếu đến quy định pháp luật nếu có.
    """


# 8. Demo truy vấn
query = "tội cưỡng ép người khác trốn đi nước ngoài hoặc ở lại nước ngoài"
docs = db.similarity_search(query, k=3)
prompt = create_prompt(query, docs)
response = generate_from_prompt(prompt)

print("\n🤖 Câu trả lời từ Gemini:")
print(response)
