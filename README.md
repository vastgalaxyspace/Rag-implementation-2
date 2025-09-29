```mermaid
flowchart TD
    A[Documents (PDF/DOCX/TXT)] --> B[Load & Read Documents]
    B --> C[Split into Chunks (RecursiveCharacterTextSplitter)]
    C --> D[Generate Embeddings (all-MiniLM-L6-v2)]
    D --> E[Store in Vector DB (ChromaDB)]

    E --> F[User enters a Question]
    F --> G[Retrieve Top-k Chunks (similarity_search)]
    G --> H[Build Prompt with Context + Question]
    H --> I[Local LLM (distilgpt2) Generate Answer]
    I --> J[Show Context & Answer]
