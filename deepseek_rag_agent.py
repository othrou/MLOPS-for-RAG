import os
import tempfile
from datetime import datetime
from typing import List
import streamlit as st
import bs4
from agno.agent import Agent
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.embeddings import Embeddings
from agno.tools.exa import ExaTools
from agno.embedder.ollama import OllamaEmbedder
from langchain_ollama import ChatOllama
from opik.integrations.langchain import OpikTracer
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

# Load environment variables
load_dotenv()

#### RAG Tracking ####
import opik
from opik import track
from opik.opik_context import update_current_trace

# Initialize Opik Tracer
opik_tracer = OpikTracer(tags=["langchain", "ollama"])

# Custom Embedder
class OllamaEmbedderr(Embeddings):
    def __init__(self, model_name="snowflake-arctic-embed"):
        self.embedder = OllamaEmbedder(id=model_name, dimensions=1024)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.embedder.get_embedding(text)

# Constants
COLLECTION_NAME = "test-deepseek-r1"

# Streamlit App Initialization
st.title("👋 Deepseek Local RAG Reasoning Agent")

# Session State Initialization
if 'qdrant_api_key' not in st.session_state:
    st.session_state.qdrant_api_key = ""
if 'qdrant_url' not in st.session_state:
    st.session_state.qdrant_url = ""
if 'model_version' not in st.session_state:
    st.session_state.model_version = "deepseek-r1:1.5b"
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = []
if 'history' not in st.session_state:
    st.session_state.history = []
if 'exa_api_key' not in st.session_state:
    st.session_state.exa_api_key = ""
if 'use_web_search' not in st.session_state:
    st.session_state.use_web_search = False
if 'force_web_search' not in st.session_state:
    st.session_state.force_web_search = False
if 'similarity_threshold' not in st.session_state:
    st.session_state.similarity_threshold = 0.7
if 'rag_enabled' not in st.session_state:
    st.session_state.rag_enabled = True

# Sidebar Configuration
st.sidebar.header("🤖 Agent Configuration")
st.sidebar.header("📚 Model Selection")
model_help = """
- 1.5b: Lighter model, suitable for most laptops
- 7b: More capable but requires better GPU/RAM
"""
st.session_state.model_version = st.sidebar.radio(
    "Select Model Version",
    options=["deepseek-r1:1.5b", "deepseek-r1:7b", "llama3.2"],
    help=model_help
)
st.sidebar.info("Run `ollama pull deepseek-r1:7b` or `deepseek-r1:1.5b` respectively.")

st.sidebar.header("⚙️ RAG Configuration")
st.session_state.rag_enabled = st.sidebar.toggle("Enable RAG Mode", value=st.session_state.rag_enabled)

if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.history = []
    st.rerun()

if st.session_state.rag_enabled:
    st.sidebar.header("🔑 API Configuration")
    st.session_state.qdrant_api_key = st.sidebar.text_input(
        "Qdrant API Key", type="password", value=st.session_state.qdrant_api_key
    )
    st.session_state.qdrant_url = st.sidebar.text_input(
        "Qdrant URL", placeholder="https://your-cluster.cloud.qdrant.io:6333", value=st.session_state.qdrant_url
    )

    st.sidebar.header("🔍 Search Configuration")
    st.session_state.similarity_threshold = st.sidebar.slider(
        "Document Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        help="Lower values will return more documents but might be less relevant."
    )

st.sidebar.header("🌐 Web Search Configuration")
st.session_state.use_web_search = st.sidebar.checkbox(
    "Enable Web Search Fallback", value=st.session_state.use_web_search
)
if st.session_state.use_web_search:
    st.session_state.exa_api_key = st.sidebar.text_input(
        "Exa AI API Key", type="password", value=st.session_state.exa_api_key
    )
    default_domains = ["arxiv.org", "wikipedia.org", "github.com", "medium.com"]
    custom_domains = st.sidebar.text_input(
        "Custom domains (comma-separated)", value=",".join(default_domains)
    )
    search_domains = [d.strip() for d in custom_domains.split(",") if d.strip()]

# Utility Functions
@track(name="track db init", capture_input=False)
def init_qdrant() -> QdrantClient | None:
    if not all([st.session_state.qdrant_api_key, st.session_state.qdrant_url]):
        return None
    try:
        return QdrantClient(
            url=st.session_state.qdrant_url,
            api_key=st.session_state.qdrant_api_key,
            timeout=60
        )
    except Exception as e:
        st.error(f"❌ Qdrant connection failed: {str(e)}")
        return None

@track(name="process_document", capture_input=True, capture_output=True)
def process_pdf(file) -> List:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            loader = PyPDFLoader(tmp_file.name)
            documents = loader.load()
            for doc in documents:
                doc.metadata.update({
                    "source_type": "pdf",
                    "file_name": file.name,
                    "timestamp": datetime.now().isoformat()
                })
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"❌ PDF processing error: {str(e)}")
        return []

@track
def process_web(url: str) -> List:
    try:
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header", "content", "main")
                )
            )
        )
        documents = loader.load()
        for doc in documents:
            doc.metadata.update({
                "source_type": "url",
                "url": url,
                "timestamp": datetime.now().isoformat()
            })
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"❌ Web processing error: {str(e)}")
        return []

@track
def create_vector_store(client, texts):
    try:
        try:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=1024,  
                    distance=Distance.COSINE
                )
            )
            st.success(f"✅ Created new collection: {COLLECTION_NAME}")
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise e
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=OllamaEmbedderr()
        )
        with st.spinner('⬆️ Uploading documents to Qdrant...'):
            vector_store.add_documents(texts)
            st.success("✅ Documents stored successfully!")
            return vector_store
    except Exception as e:
        st.error(f"❌ Vector store error: {str(e)}")
        return None

@track(name="web_search", capture_input=True)
def get_web_search_agent() -> Agent:
    return Agent(
        name="Web Search Agent",
        model=ChatOllama(
            model="llama3.2",
            temperature=0.3,
        ).with_config({"callbacks": [opik_tracer]}),
        tools=[ExaTools(
            api_key=st.session_state.exa_api_key,
            include_domains=search_domains,
            num_results=5
        )],
        instructions="""You are a web search expert. Your task is to:
        1. Search the web for relevant information about the query
        2. Compile and summarize the most relevant information
        3. Include sources in your response""",
        show_tool_calls=True,
        markdown=True,
    )

@track
def get_rag_agent() -> Agent:
    return Agent(
        name="DeepSeek RAG Agent",
        model=ChatOllama(
            model=st.session_state.model_version,
            temperature=0
        ).with_config({"callbacks": [opik_tracer]}),
        instructions="""You are an Intelligent Agent specializing in providing accurate answers.
        When asked a question:
        - Analyze the question and answer the question with what you know.
        When given context from documents:
        - Focus on information from the provided documents
        - Be precise and cite specific details
        When given web search results:
        - Clearly indicate that the information comes from web search
        - Synthesize the information clearly
        Always maintain high accuracy and clarity in your responses.""",
        show_tool_calls=True,
        markdown=True,
    )

@track(name="check_document_relevance", capture_input=True)
def check_document_relevance(query: str, vector_store, threshold: float = 0.7) -> tuple[bool, List]:
    if not vector_store:
        return False, []
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": threshold}
    )
    docs = retriever.invoke(query)
    return bool(docs), docs

# Chat Input and Toggle
chat_col, toggle_col = st.columns([0.9, 0.1])
with chat_col:
    prompt = st.chat_input("Ask about your documents..." if st.session_state.rag_enabled else "Ask me anything...")
with toggle_col:
    st.session_state.force_web_search = st.toggle('🌐', help="Force web search")

# Main Logic
if st.session_state.rag_enabled:
    qdrant_client = init_qdrant()
    st.sidebar.header("📂 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    web_url = st.sidebar.text_input("Or enter URL")

    # Process Uploaded Documents
    if uploaded_file:
        file_name = uploaded_file.name
        if file_name not in st.session_state.processed_documents:
            with st.spinner('🔄 Processing PDF...'):
                texts = process_pdf(uploaded_file)
                if texts and qdrant_client:
                    if st.session_state.vector_store:
                        st.session_state.vector_store.add_documents(texts)
                    else:
                        st.session_state.vector_store = create_vector_store(qdrant_client, texts)
                    st.session_state.processed_documents.append(file_name)
                    st.success(f"✅ Added PDF: {file_name}")

    if web_url:
        if web_url not in st.session_state.processed_documents:
            with st.spinner('🔄 Processing URL...'):
                texts = process_web(web_url)
                if texts and qdrant_client:
                    if st.session_state.vector_store:
                        st.session_state.vector_store.add_documents(texts)
                    else:
                        st.session_state.vector_store = create_vector_store(qdrant_client, texts)
                    st.session_state.processed_documents.append(web_url)
                    st.success(f"✅ Added URL: {web_url}")

    # Display Processed Sources
    if st.session_state.processed_documents:
        st.sidebar.header("📖 Processed Sources")
        for source in st.session_state.processed_documents:
            icon = "📄" if source.endswith('.pdf') else "🌐"
            st.sidebar.text(f"{icon} {source}")

if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if st.session_state.rag_enabled:
        with st.spinner("🧠 Evaluating the Query..."):
            try:
                rewritten_query = prompt
                with st.expander("Evaluating the query"):
                    st.write(f"User's Prompt: {prompt}")
            except Exception as e:
                st.error(f"❌ Error rewriting query: {str(e)}")
                rewritten_query = prompt

        context = ""
        docs = []
        if not st.session_state.force_web_search and st.session_state.vector_store:
            retriever = st.session_state.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 5,
                    "score_threshold": st.session_state.similarity_threshold
                }
            )
            docs = retriever.invoke(rewritten_query)
            if docs:
                context = "\n\n".join([d.page_content for d in docs])
                st.info(f"📚 Found {len(docs)} relevant documents (similarity > {st.session_state.similarity_threshold})")
            elif st.session_state.use_web_search:
                st.info("🔄 No relevant documents found in database, falling back to web search...")

        if (st.session_state.force_web_search or not context) and st.session_state.use_web_search and st.session_state.exa_api_key:
            with st.spinner("🌐 Searching the web..."):
                try:
                    web_search_agent = get_web_search_agent()
                    web_results = web_search_agent.invoke(
                        {"input": rewritten_query},
                        config=RunnableConfig(callbacks=[opik_tracer])
                    ).content
                    if web_results:
                        context = f"Web Search Results:\n{web_results}"
                        if st.session_state.force_web_search:
                            st.info("ℹ️ Using web search as requested via toggle.")
                        else:
                            st.info("ℹ️ Using web search as fallback since no relevant documents were found.")
                except Exception as e:
                    st.error(f"❌ Web search error: {str(e)}")

        with st.spinner("🤖 Thinking..."):
            try:
                rag_agent = get_rag_agent()
                if context:
                    full_prompt = f"""Context: {context}

Original Question: {prompt}
Please provide a comprehensive answer based on the available information."""
                else:
                    full_prompt = f"Original Question: {prompt}\n"
                    st.info("ℹ️ No relevant information found in documents or web search.")

                response = rag_agent.invoke(
                    {"input": full_prompt},
                    config=RunnableConfig(callbacks=[opik_tracer])
                )
                st.session_state.history.append({
                    "role": "assistant",
                    "content": response.content
                })

                with st.chat_message("assistant"):
                    st.write(response.content)
                    if not st.session_state.force_web_search and docs:
                        with st.expander("🔍 See document sources"):
                            for i, doc in enumerate(docs, 1):
                                source_type = doc.metadata.get("source_type", "unknown")
                                source_icon = "📄" if source_type == "pdf" else "🌐"
                                source_name = doc.metadata.get("file_name" if source_type == "pdf" else "url", "unknown")
                                st.write(f"{source_icon} Source {i} from {source_name}:")
                                st.write(f"{doc.page_content[:200]}...")
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
    else:
        with st.spinner("🤖 Thinking..."):
            try:
                rag_agent = get_rag_agent()
                web_search_agent = get_web_search_agent() if st.session_state.use_web_search else None
                context = ""
                if st.session_state.force_web_search and web_search_agent:
                    with st.spinner("🌐 Searching the web..."):
                        try:
                            web_results = web_search_agent.invoke(
                                {"input": prompt},
                                config=RunnableConfig(callbacks=[opik_tracer])
                            ).content
                            if web_results:
                                context = f"Web Search Results:\n{web_results}"
                                st.info("ℹ️ Using web search as requested.")
                        except Exception as e:
                            st.error(f"❌ Web search error: {str(e)}")

                if context:
                    full_prompt = f"""Context: {context}

Question: {prompt}
Please provide a comprehensive answer based on the available information."""
                else:
                    full_prompt = prompt

                response = rag_agent.invoke(
                    {"input": full_prompt},
                    config=RunnableConfig(callbacks=[opik_tracer])
                )

                import re
                think_pattern = r'<think>(.*?)</think>'
                think_match = re.search(think_pattern, response.content, re.DOTALL)
                if think_match:
                    thinking_process = think_match.group(1).strip()
                    final_response = re.sub(think_pattern, '', response.content, flags=re.DOTALL).strip()
                else:
                    thinking_process = None
                    final_response = response.content

                st.session_state.history.append({
                    "role": "assistant",
                    "content": final_response
                })

                with st.chat_message("assistant"):
                    if thinking_process:
                        with st.expander("🧠 See thinking process"):
                            st.markdown(thinking_process)
                    st.markdown(final_response)
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
else:
    st.warning("You can directly talk to r1 locally! Toggle the RAG mode to upload documents!")