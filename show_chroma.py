import chromadb

# ============================================================
# CONNECT TO EXISTING CHROMADB
# ============================================================

DB_PATH = "./chroma_db"

client = chromadb.PersistentClient(path=DB_PATH)

print("=" * 70)
print("                 CHROMADB INSPECTION")
print("=" * 70)

print("\n📁 ChromaDB Location:")
print(DB_PATH)


# ============================================================
# 1. SHOW COLLECTIONS
# ============================================================

collections = client.list_collections()

print("\n" + "=" * 70)
print("1. COLLECTIONS")
print("=" * 70)

if not collections:
    print("❌ No collections found.")
    exit()

for collection in collections:
    print(f"Collection Name : {collection.name}")


# ============================================================
# SELECT COLLECTION
# ============================================================

collection = collections[0]

print("\nSelected Collection:")
print(collection.name)


# ============================================================
# 2. SHOW NUMBER OF STORED CHUNKS
# ============================================================

count = collection.count()

print("\n" + "=" * 70)
print("2. NUMBER OF STORED CHUNKS")
print("=" * 70)

print(f"Total chunks stored in ChromaDB: {count}")


# ============================================================
# 3. GET DOCUMENTS + METADATA + EMBEDDINGS
# ============================================================

data = collection.get(
    include=["documents", "metadatas", "embeddings"]
)

documents = data.get("documents")
metadatas = data.get("metadatas")
embeddings = data.get("embeddings")
ids = data.get("ids")


# ============================================================
# 3. STORED DOCUMENT CHUNKS
# ============================================================

print("\n" + "=" * 70)
print("3. STORED DOCUMENT CHUNKS")
print("=" * 70)

if documents:

    show_count = min(5, len(documents))

    for i in range(show_count):

        print(f"\n--- CHUNK {i + 1} ---")

        print("\nID:")
        print(ids[i])

        print("\nDocument Text:")
        print(documents[i][:1000])

        if len(documents[i]) > 1000:
            print("\n... [text shortened for display]")


# ============================================================
# 4. METADATA
# ============================================================

print("\n" + "=" * 70)
print("4. METADATA")
print("=" * 70)

if metadatas:

    for i in range(min(5, len(metadatas))):

        print(f"\n--- CHUNK {i + 1} METADATA ---")

        print(metadatas[i])


# ============================================================
# 5. EMBEDDINGS
# ============================================================

print("\n" + "=" * 70)
print("5. EMBEDDINGS")
print("=" * 70)

if embeddings is not None:

    print("✅ Embeddings are stored in ChromaDB.")

    print("\nNumber of embedding vectors:")
    print(len(embeddings))

    first_embedding = embeddings[0]

    print("\nEmbedding dimension:")
    print(len(first_embedding))

    print("\nFirst 20 values of the first embedding:")
    print(first_embedding[:20])

    print("\nFull embedding:")
    print(first_embedding)

else:

    print("❌ No embeddings were returned.")


# ============================================================
# 6. CHROMADB INTERNAL FILES
# ============================================================

print("\n" + "=" * 70)
print("6. CHROMADB INTERNAL FILES")
print("=" * 70)

print("""
chroma_db/
│
├── chroma.sqlite3
│      → Database file used by ChromaDB
│
└── UUID folder/
       │
       ├── data_level0.bin
       │      → Internal vector index data
       │
       ├── header.bin
       │      → Index information
       │
       ├── length.bin
       │      → Internal index information
       │
       └── link_lists.bin
              → Vector-search index information

These files are automatically managed by ChromaDB.
They should NOT be manually edited.
""")


# ============================================================
# 7. COMPLETE RAG FLOW
# ============================================================

print("\n" + "=" * 70)
print("7. HOW CHROMADB FITS INTO THE RAG PIPELINE")
print("=" * 70)

print("""
University Documents
        ↓
Document Loading
        ↓
Chunking
        ↓
Embedding Model
        ↓
Numerical Vectors
        ↓
      ChromaDB
        ↓
   Store + Search
        ↓
User Question
        ↓
Question Embedding
        ↓
Similarity Search
        ↓
Top Relevant Chunks
        ↓
      LlamaIndex
        ↓
   Groq API / LLM
        ↓
    Final Answer
""")


# ============================================================
# 8. FINAL SUMMARY
# ============================================================

print("=" * 70)
print("8. SUMMARY")
print("=" * 70)

print(f"""
Collection Name : {collection.name}
Stored Chunks   : {count}

ChromaDB is responsible for:
✓ Storing document chunks
✓ Storing embeddings
✓ Storing metadata
✓ Performing similarity search

The .bin and .sqlite3 files are internal ChromaDB storage files.
They are accessed through the ChromaDB API rather than opened manually.
""")

print("=" * 70)
print("          CHROMADB INSPECTION COMPLETE")
print("=" * 70)