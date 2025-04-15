import os
import json
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
from src.rag.Multi_Agent import get_rag_agent
from data.eval_dataset import create_eval_ds 
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    load_dotenv()
    
    # Validate environment variables
    required_vars = ["GOOGLE_API_KEY", "qdrant_url", "qdrant_api_key"]
    for var in required_vars:
        if not os.environ.get(var):
            raise ValueError(f"{var} environment variable is missing.")

    logger.info("Initializing components...")
    
    # Initialize components
    eval_llm = GoogleGenAI(
        model="gemini-2.0-flash",
        api_key=os.environ.get("GOOGLE_API_KEY"),
    )

    q_client = qdrant_client.QdrantClient(
        url=os.environ.get('qdrant_url'),
        api_key=os.environ.get('qdrant_api_key')
    )

    vector_db = Qdrant(
        collection="rag_test",
        url=os.environ.get('qdrant_url'),
        api_key=os.environ.get('qdrant_api_key'),
        embedder=OllamaEmbedder(id="snowflake-arctic-embed", dimensions=1024)
    )

    knowledge_base = PDFKnowledgeBase(
        vector_db=vector_db,
        path="data/test_data.pdf",
        chunking_strategy=FixedSizeChunking(chunk_size=2000, overlap=200)
    )

    if not q_client.collection_exists(collection_name="rag_test"):
        knowledge_base.load(recreate=False)

    agent = get_rag_agent()
    eval_dataset = create_eval_ds(agent=agent, ground_truth_path="data/eval_data.json")

    # Run evaluation
    logger.info("Starting evaluation...")
    evaluation_dataset = EvaluationDataset.from_list(eval_dataset)
    evaluator_llm = LlamaIndexLLMWrapper(llm=eval_llm)
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[
            Faithfulness(),
            ContextRelevance(),
            ContextUtilization(),
            ContextRecall(),
            FactualCorrectness()
        ]
    )

    # Generate CML report
    report = {
        "summary": "RAG Evaluation Results",
        "metrics": {},
        "details": []
    }

    for metric, score in zip(result.metrics, result.scores):
        report["metrics"][metric.name] = float(score)
        report["details"].append({
            "metric": metric.name,
            "score": float(score),
            "description": str(metric)
        })

    # Write report to file
    with open("cml_report.txt", "w") as f:
        f.write(f"# RAG Evaluation Report\n\n")
        f.write(f"## Summary Metrics\n")
        for metric, score in report["metrics"].items():
            f.write(f"- **{metric}**: {score:.3f}\n")
        
        f.write("\n## Detailed Results\n")
        for detail in report["details"]:
            f.write(f"### {detail['metric']}\n")
            f.write(f"Score: {detail['score']:.3f}\n")
            f.write(f"Description: {detail['description']}\n\n")

    logger.info("Evaluation completed. Report generated.")

    # Cleanup
    if q_client.collection_exists(collection_name="rag_test"):
        q_client.delete_collection(collection_name="rag_test")

if __name__ == "__main__":
    main()
