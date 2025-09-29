from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ---------------------------
# Step 1: Load vector store
# ---------------------------
print("Loading vector database...")
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_function
)
print(f"Vector database loaded successfully!")

# ---------------------------
# Step 2: Load local LLM
# ---------------------------
model_name = "distilgpt2"  # Using a lightweight text generation model
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# ---------------------------
# Step 3: Query Loop
# ---------------------------
while True:
    query = input("\nEnter your question (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break

    # Retrieve top 5 relevant chunks
    docs = vectordb.similarity_search(query, k=5)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Prepare prompt
    prompt = f"Answer the question based ONLY on the context below:\n\n{context}\n\nQuestion: {query}\nAnswer:"

    # Generate answer
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True, padding=True)
    
    with torch.no_grad():
        output_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2
        )
    
    # Decode only the new tokens (not the input prompt)
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print("\nRetrieved Context:")
    print(context[:300] + "..." if len(context) > 300 else context)
    print("\nGenerated Answer:")
    print(answer if answer.strip() else "No clear answer found in the context.")
