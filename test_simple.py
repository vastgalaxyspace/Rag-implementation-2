print("Testing imports...")

try:
    from langchain_community.vectorstores import Chroma
    print("✅ Chroma import successful")
except ImportError as e:
    print(f"❌ Chroma import failed: {e}")

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("✅ HuggingFaceEmbeddings import successful")
except ImportError as e:
    print(f"❌ HuggingFaceEmbeddings import failed: {e}")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("✅ Transformers import successful")
except ImportError as e:
    print(f"❌ Transformers import failed: {e}")

print("\nTesting vector database...")
try:
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="chroma_db", embedding_function=embedding_function)
    
    # Test a simple query
    docs = vectordb.similarity_search("hallucinate", k=2)
    print(f"✅ Found {len(docs)} documents")
    if docs:
        print(f"Sample content: {docs[0].page_content[:200]}...")
except Exception as e:
    print(f"❌ Vector database test failed: {e}")

print("Test complete!")