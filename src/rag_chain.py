# src/rag_chain.py
# LCEL RAG chain: retriever → prompt → Gemini → answer

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
import sys
from pathlib import Path

# Allow `python src/rag_chain.py` as well as `python -m src.rag_chain`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retriever import get_retriever  # noqa: E402

load_dotenv()

PROMPT = ChatPromptTemplate.from_template("""
You are an AI engineering knowledge assistant.
Answer the question using ONLY the context provided. Do not start your answer with "Based on the provided context" or similar phrases — answer directly.
If the context doesn't contain the answer, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_chain():
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1
    )
    retriever = get_retriever(k=4)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


if __name__ == "__main__":
    chain = build_chain()
    questions = [
        "What is retrieval-augmented generation?",
        "How does RAGAS evaluate RAG systems?",
        "What is prompt engineering?",
        "What is the capital of France?", # Testing out of scope
        "Who is Elon Musk?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        answer = chain.invoke(q)
        print(f"A: {answer}")
        print("-" * 60)


def ask_with_sources(question: str, k: int = 4):
    """Return (answer, source_docs) for a given question."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1
    )
    retriever = get_retriever(k=k)
    docs = retriever.invoke(question)
    context = format_docs(docs)

    prompt_text = PROMPT.format_messages(
        context=context,
        question=question
    )
    response = llm.invoke(prompt_text)
    # gemini-3.5 may return content as a list of parts; keep text only
    if isinstance(response.content, list):
        answer = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in response.content
        )
    else:
        answer = response.content
    return answer, docs
