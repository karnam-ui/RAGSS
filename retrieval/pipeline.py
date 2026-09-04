import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline

from retrieval.retrieve_llm_pipeline import retrieve_llm_pipeline
from retrieval.retriver import Retriever

llm = HuggingFacePipeline(
    pipeline=retrieve_llm_pipeline,
)

class BasicRetriever:
    def __init__(self, retriever):
        self.retriever = retriever

        self.prompt = ChatPromptTemplate.from_template("""
You are a Project Guide assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Answer ONLY what the user asked.
- Do not add unrelated information from the context.
- Do not repeat information.
- If the question asks for a list, provide only the requested list.
- If the context contains unrelated projects, topics, or text, ignore them.
- Do not invent or infer information that is not explicitly supported by the context.
- Keep the answer concise and direct.
- If the context does not contain enough information to answer the question, say:
  "The provided documents do not contain enough information to answer this."

Context:
{context}

Question:
{question}

""")

        # prompt → LLM
        self.chain = self.prompt | llm

    def generate(self, query):
        results = self.retriever.retrieve(query)

        # Keep metadata out of the prompt so the model receives only source text.
        context = "\n\n".join(
            result.page_content
            for result in results
        )

        response = self.chain.invoke({
            "context": context,
            "question": query
        })

        citations = [
            {
                "source": result.metadata.get("source"),
                "title": result.metadata.get("title"),
                "page": int(result.metadata.get("page", 0)) + 1,
                "total_pages": result.metadata.get("total_pages"),
            }
            for result in results
        ]

        return {
            "answer": response,
            "citations": citations,
        }

