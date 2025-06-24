import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
res = embedding.embed_query("Giấc ngủ là gì?")
print(res[:10])
