# src/ingest.py
# Chunk corpus docs → embed with Voyage AI → upsert to Supabase pgvector

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client

load_dotenv()

# --- Config ---
DOCS_DIR = Path(__file__).parent.parent / "data" / "docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def main():
    # --- Clients ---
    supabase = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_KEY")
    )

    embeddings = VoyageAIEmbeddings(
        voyage_api_key=os.getenv("VOYAGE_API_KEY"),
        model="voyage-3"  # 1024 dims, matches documents.embedding vector(1024)
    )

    # --- Load docs ---
    loader = DirectoryLoader(
        str(DOCS_DIR),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents")

    if not docs:
        print(f"No .txt files found in {DOCS_DIR}. Add corpus files first.")
        return

    # --- Split ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    # --- Clear existing rows so reruns don't duplicate ---
    supabase.table("documents").delete().gte("id", 0).execute()
    print("Cleared existing rows in documents")

    # --- Upsert to Supabase ---
    print("Embedding + upserting to Supabase...")
    vector_store = SupabaseVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=supabase,
        table_name="documents",
        query_name="match_documents"
    )

    print(f"✓ Ingested {len(chunks)} chunks into Supabase")

if __name__ == "__main__":
    main()
