import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

pdf_file = os.path.join("data", "Manual.pdf")
loader = PyPDFLoader(pdf_file)
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", r"\. ", " ", ""]
)
document_chunks = splitter.split_documents(pages)

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

output_path = "cisco_7604_index"
vector_store = FAISS.from_documents(document_chunks, embeddings)
vector_store.save_local(output_path)

bm25_retriever = BM25Retriever.from_documents(document_chunks)
faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

def hybrid_retrieve(query: str, max_results: int = 5):
    bm25_docs = bm25_retriever.invoke(query)
    faiss_docs = faiss_retriever.invoke(query)

    seen_content = set()
    combined = []
    for doc in (bm25_docs + faiss_docs):
        if doc.page_content not in seen_content:
            seen_content.add(doc.page_content)
            combined.append(doc)
        if len(combined) >= max_results:
            break
    
    return combined

print(f"Ingestion complete! Created {len(document_chunks)} chunks in {output_path}")