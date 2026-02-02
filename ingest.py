import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

load_dotenv()

pdf_file = os.path.join("data", "Manual.pdf")
loader = PyPDFLoader(pdf_file)
pages = loader.load()

# Voice-optimized chunking (smaller, sentence-friendly)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_documents(pages)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

index_path = "cisco_7604_index"
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local(index_path)

bm25 = BM25Retriever.from_documents(chunks)

print(f"Ingestion complete: {len(chunks)} voice-optimized chunks saved.")
