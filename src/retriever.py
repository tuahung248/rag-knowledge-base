# src/retriever.py
# Query Supabase pgvector — returns top-k relevant chunks

import os
from dotenv import load_dotenv
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client

load_dotenv()

def get_retriever(k: int = 4):
    supabase = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_KEY")
    )
    embeddings = VoyageAIEmbeddings(
        voyage_api_key=os.getenv("VOYAGE_API_KEY"),
        model="voyage-3"  # must match ingest.py model
    )
    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents"
    )
    return vector_store.as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    # Smoke test
    retriever = get_retriever()
    query = "What is RAG and how does it work?"
    results = retriever.invoke(query)
    print(f"Query: {query}")
    print(f"Top {len(results)} results:\n")
    for i, doc in enumerate(results):
        print(f"[{i+1}] {doc.page_content[:200]}\n")
