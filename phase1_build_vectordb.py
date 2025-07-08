import os
import glob
from dotenv import load_dotenv

from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 1. Load biến môi trường
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise EnvironmentError("❌ Thiếu GOOGLE_API_KEY trong file .env")

# 2. Load tài liệu từ các file .docx trong thư mục "data"
all_documents = []
file_paths = glob.glob("data/*.docx")
for path in file_paths:
    loader = Docx2txtLoader(path)
    all_documents.extend(loader.load())
print(
    f"✅ Đã load {len(file_paths)} file .docx với tổng cộng {len(all_documents)} đoạn văn")

# 3. Chia nhỏ văn bản thành các đoạn để tạo embedding
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=120)
chunks = splitter.split_documents(all_documents)
print(f"✅ Đã chia thành {len(chunks)} đoạn nhỏ để đưa vào vector DB")

# 4. Tạo embedding bằng Gemini
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 5. Tạo FAISS Vector Store và lưu lại
db = FAISS.from_documents(chunks, embedding)
db.save_local("faiss_yhgn_db")
print("✅ Đã tạo và lưu FAISS Vector Store thành công tại thư mục: faiss_yhgn_db")
