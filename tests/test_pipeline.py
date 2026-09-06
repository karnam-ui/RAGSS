import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
   sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.pipeline import BasicRetriever
from retrieval.retriver import Retriever

answerer = BasicRetriever(Retriever())
res = answerer.generate("What is syllabus in the unit 2?")
print("Answer:\n", res["answer"])
print("\nCitations:")
for citation in res["citations"]:
   print(citation)