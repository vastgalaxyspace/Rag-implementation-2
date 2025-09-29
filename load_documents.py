import os
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma

# Folder containing your documents
DOCUMENTS_FOLDER = "data_folder"

def load_pdf(file_path):
    text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def load_docx(file_path):
    text = ""
    doc = Document(file_path)
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def load_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            documents.append(load_pdf(file_path))
        elif filename.endswith(".docx"):
            documents.append(load_docx(file_path))
        elif filename.endswith(".txt"):
            documents.append(load_txt(file_path))
    return documents


if __name__ == "__main__":
    # Load documents first
    documents = load_documents(DOCUMENTS_FOLDER)
    print(f"Loaded {len(documents)} documents.")
    
    # ---------------------------
    # Step 2: Split documents into chunks
    # ---------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,   # 500 characters per chunk
        chunk_overlap=50  # overlap to keep context
    )

    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_text(doc)
        all_chunks.extend(chunks)

    print(f"Total chunks created: {len(all_chunks)}")
    print("Sample chunk:")
    try:
        print(all_chunks[0][:300])
    except UnicodeEncodeError:
        print(all_chunks[0][:300].encode('utf-8', errors='ignore').decode('utf-8'))

    # ---------------------------
    # Step 3: Generate embeddings
    # ---------------------------
    from langchain.embeddings import HuggingFaceEmbeddings
    
    # Create embedding function compatible with Chroma
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # ---------------------------
    # Step 4: Store embeddings in Chroma
    # ---------------------------
    persist_directory = "chroma_db"  # folder to save vector DB

    vectordb = Chroma.from_texts(
        texts=all_chunks,
        embedding=embedding_function,
        persist_directory=persist_directory
    )

    vectordb.persist()
    print(f"Vector store saved at {persist_directory}")
    
    # Print first 1000 characters of the first document as a sample
    if documents:
        print("Sample document content:")
        try:
            print(documents[0][:1000])
        except UnicodeEncodeError:
            print(documents[0][:1000].encode('utf-8', errors='ignore').decode('utf-8'))