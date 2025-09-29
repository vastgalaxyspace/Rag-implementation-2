```mermaid
flowchart TD
    A[Documents (PDF/DOCX/TXT)] --> B[Load & Read Documents]
    B --> C[Split into Chunks<br/>(RecursiveCharacterTextSplitter)]
    C --> D[Generate Embeddings<br/>(all-MiniLM-L6-v2)]
    D --> E[Store in Vector DB<br/>(ChromaDB)]

    E --> F[User enters a Question]
    F --> G[Retrieve Top-k Chunks<br/>(similarity_search)]
    G --> H[Build Prompt with<br/>Context + Question]
    H --> I[Local LLM (distilgpt2)<br/>Generate Answer]
    I --> J[Show Context & Answer]
