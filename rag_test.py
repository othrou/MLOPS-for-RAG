import os
from agno.agent import Agent
from agno.embedder.ollama import OllamaEmbedder
from qdrant_client import qdrant_client
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.vectordb.qdrant import Qdrant
from agno.document.chunking.fixed import FixedSizeChunking
from dotenv import load_dotenv, find_dotenv
from ragas.llms import LlamaIndexLLMWrapper
from ragas import EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, FactualCorrectness, ContextRelevance, ContextUtilization, ContextRecall
from llama_index.llms.google_genai import GoogleGenAI

#from src.rag.embedding import OllamaEmbedderr
from src.rag.Multi_Agent import get_rag_agent
from data.eval_dataset import create_eval_ds 

import logging

logging.basicConfig(level=logging.INFO)

load_dotenv()

# this is a test file, so i added this 2 lines to ensure that there's no problem with environment variables
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable is missing.")
if not os.environ.get("qdrant_url") or not os.environ.get("qdrant_api_key"):
    raise ValueError("Qdrant URL or API key is missing.")


logging.info("Initializing LLM...")

eval_llm = GoogleGenAI(  
    model="gemini-2.0-flash",
     api_key=os.environ.get("GOOGLE_API_KEY"),
)


doc_path = "data/test_data.pdf"
ground_truth_path = "data/eval_data.json"


# initialize the qdrant client.
q_client = qdrant_client.QdrantClient(url=os.environ.get('qdrant_url'), api_key=os.environ.get('qdrant_api_key'))

# create the qdrant vector store instance
vector_db = Qdrant(
    collection="rag_test",
    url=os.environ.get('qdrant_url'),
    api_key=os.environ.get('qdrant_api_key'),
    embedder=OllamaEmbedder(id="snowflake-arctic-embed",dimensions=1024)

)


# configure the knowledge base
knowledge_base = PDFKnowledgeBase(vector_db=vector_db,
                                  path=doc_path,
                                  chunking_strategy=FixedSizeChunking(
                                      chunk_size=2000,
                                      overlap=200)
                                  )

if not q_client.collection_exists(collection_name=os.environ.get('collection_name')):
    knowledge_base.load(recreate=False)

# initialize agent
agent = get_rag_agent()

# create the dataset for evaluation
eval_dataset = create_eval_ds(agent=agent, ground_truth_path=ground_truth_path)

# trigger evals
evaluation_dataset = EvaluationDataset.from_list(eval_dataset)
evaluator_llm = LlamaIndexLLMWrapper(llm=eval_llm)
result = evaluate(dataset=evaluation_dataset, metrics=[Faithfulness(), ContextRelevance(),
                                                       ContextUtilization(), ContextRecall(),
                                                       FactualCorrectness()])

for score in result.scores:
    print(score)

# destroy the collection
if q_client.collection_exists(collection_name=os.environ.get('collection_name')):
    q_client.delete_collection(collection_name=os.environ.get('collection_name'))