from langtrace_python_sdk import langtrace
from langtrace_python_sdk.utils.with_root_span import with_langtrace_root_span
import os
import streamlit as st
from src.rag.embedding import OllamaEmbedder
from src.rag.database import init_qdrant, create_vector_store
from src.rag.processing import process_pdf, process_web, check_document_relevance
from src.rag.Multi_Agent import get_web_search_agent, get_rag_agent
from dotenv import find_dotenv, load_dotenv

import hydra
import logging
from omegaconf.omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra


logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Langtrace
langtrace.init(api_key=os.getenv("LANGTRACE_API_KEY"))


@hydra.main(config_path="./config", config_name="config",version_base=None)
def main(cfg):

    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    logger.info(f"Using the model: {cfg.model.COLLECTION_NAME}")
    
    # Constants
    COLLECTION_NAME = cfg.model.COLLECTION_NAME


    # Streamlit App Initialization
    st.title("🐋 Deepseek Local RAG Reasoning Agent")

    # Session State Initialization
    if 'google_api_key' not in st.session_state:
        st.session_state.google_api_key = ""
    if 'qdrant_api_key' not in st.session_state:
        st.session_state.qdrant_api_key = ""
    if 'qdrant_url' not in st.session_state:
        st.session_state.qdrant_url = ""
    if 'model_version' not in st.session_state:
        st.session_state.model_version = cfg.model.model  # default model version which is the deepseek-r1:1.5b
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
        st.session_state.similarity_threshold = cfg.processing.similarity.threshold  # default threshold which is 0.7
    if 'rag_enabled' not in st.session_state:
        st.session_state.rag_enabled = True  # RAG is enabled by default


    # Sidebar Configuration
    st.sidebar.header("🤖 Agent Configuration")

    # Model Selection
    st.sidebar.header("📦 Model Selection")
    model_help = """
    - 1.5b: Lighter model, suitable for most laptops
    - 7b: More capable but requires better GPU/RAM

    Choose based on your hardware capabilities.
    """


    st.session_state.model_version = st.sidebar.selectbox(
        "Select Model Version",
        options=list(cfg.model.available_models),  # dtype should be an integer index
        help=model_help
    )
    st.sidebar.info("Run ollama pull deepseek-r1:1.5b, gemma3:1b...")

    # RAG Mode Toggle
    st.sidebar.header("🔍 RAG Configuration")
    st.session_state.rag_enabled = st.sidebar.toggle("Enable RAG Mode", value=st.session_state.rag_enabled)

    # Clear Chat Button
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.history = []
        st.rerun()

    # Show API Configuration only if RAG is enabled
    if st.session_state.rag_enabled:
        st.sidebar.header("🔑 API Configuration")
        qdrant_api_key = st.sidebar.text_input("Qdrant API Key", type="password", value=st.session_state.qdrant_api_key)
        qdrant_url = st.sidebar.text_input("Qdrant URL", 
                                        placeholder="https://your-cluster.cloud.qdrant.io:6333",
                                        value=st.session_state.qdrant_url)

        # Update session state
        st.session_state.qdrant_api_key = qdrant_api_key
        st.session_state.qdrant_url = qdrant_url
        
        # Search Configuration (only shown in RAG mode)
        st.sidebar.header("🎯 Search Configuration")
        st.session_state.similarity_threshold = st.sidebar.slider(
            "Document Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=cfg.processing.similarity.threshold,  # default threshold which is 0.7
            help="Lower values will return more documents but might be less relevant. Higher values are more strict."
        )

    # Add in the sidebar configuration section, after the existing API inputs

    st.sidebar.header("🌐 Web Search Configuration")
    st.session_state.use_web_search = st.sidebar.checkbox("Enable Web Search Fallback", value=st.session_state.use_web_search)

    if st.session_state.use_web_search:
        exa_api_key = st.sidebar.text_input(
            "Exa AI API Key", 
            type="password",
            value=st.session_state.exa_api_key,
            help="Required for web search fallback when no relevant documents are found"
        )
        st.session_state.exa_api_key = exa_api_key
        
        # Optional domain filtering
        default_domains = cfg.web_search.custom_domains
        custom_domains = st.sidebar.text_input(
            "Custom domains (comma-separated)", 
            value=",".join(default_domains),
            help="Enter domains to search from, e.g.: arxiv.org,wikipedia.org"
        )
        search_domains = [d.strip() for d in custom_domains.split(",") if d.strip()]

    # Search Configuration moved inside RAG mode check

    chat_col, toggle_col = st.columns([0.9, 0.1])

    with chat_col:
        prompt = st.chat_input("Ask about your documents..." if st.session_state.rag_enabled else "Ask me anything...")

    with toggle_col:
        st.session_state.force_web_search = st.toggle('🌐', help="Force web search")

    # Check if RAG is enabled 
    if st.session_state.rag_enabled:
        qdrant_client = init_qdrant()
        
        # File/URL Upload Section
        st.sidebar.header("📁 Data Upload")
        uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
        web_url = st.sidebar.text_input("Or enter URL")
        
        # Process documents
        if uploaded_file:
            file_name = uploaded_file.name
            if file_name not in st.session_state.processed_documents:
                with st.spinner('Processing PDF...'):
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
                with st.spinner('Processing URL...'):
                    texts = process_web(web_url)
                    if texts and qdrant_client:
                        if st.session_state.vector_store:
                            st.session_state.vector_store.add_documents(texts)
                        else:
                            st.session_state.vector_store = create_vector_store(qdrant_client, texts)
                        st.session_state.processed_documents.append(web_url)
                        st.success(f"✅ Added URL: {web_url}")

        # Display sources in sidebar
        if st.session_state.processed_documents:
            st.sidebar.header("📚 Processed Sources")
            for source in st.session_state.processed_documents:
                if source.endswith('.pdf'):
                    st.sidebar.text(f"📄 {source}")
                else:
                    st.sidebar.text(f"🌐 {source}")

    if prompt:
        # Add user message to history
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        if st.session_state.rag_enabled:

                # Existing RAG flow remains unchanged
                with st.spinner("🤔Evaluating the Query..."):
                    try:
                        rewritten_query = prompt
                        
                        with st.expander("Evaluating the query"):
                            st.write(f"User's Prompt: {prompt}")
                    except Exception as e:
                        st.error(f"❌ Error rewriting query: {str(e)}")
                        rewritten_query = prompt

                # Step 2: Choose search strategy based on force_web_search toggle
                context = ""
                docs = []
                if not st.session_state.force_web_search and st.session_state.vector_store:
                    # Try document search first
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
                        st.info(f"📊 Found {len(docs)} relevant documents (similarity > {st.session_state.similarity_threshold})")
                    elif st.session_state.use_web_search:
                        st.info("🔄 No relevant documents found in database, falling back to web search...")

                # Step 3: Use web search if:
                # 1. Web search is forced ON via toggle, or
                # 2. No relevant documents found AND web search is enabled in settings
                if (st.session_state.force_web_search or not context) and st.session_state.use_web_search and st.session_state.exa_api_key:
                    with st.spinner("🔍 Searching the web..."):
                        try:
                            web_search_agent = get_web_search_agent()
                            web_results = web_search_agent.run(rewritten_query).content
                            if web_results:
                                context = f"Web Search Results:\n{web_results}"
                                if st.session_state.force_web_search:
                                    st.info("ℹ️ Using web search as requested via toggle.")
                                else:
                                    st.info("ℹ️ Using web search as fallback since no relevant documents were found.")
                        except Exception as e:
                            st.error(f"❌ Web search error: {str(e)}")

                # Step 4: Generate response using the RAG agent
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

                        response = rag_agent.run(full_prompt)
                        
                        # Add assistant response to history
                        st.session_state.history.append({
                            "role": "assistant",
                            "content": response.content
                        })
                        
                        # Display assistant response
                        with st.chat_message("assistant"):
                            st.write(response.content)
                            
                            # Show sources if available
                            if not st.session_state.force_web_search and 'docs' in locals() and docs:
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
            # Simple mode without RAG
            with st.spinner("🤖 Thinking..."):
                try:
                    rag_agent = get_rag_agent()
                    web_search_agent = get_web_search_agent() if st.session_state.use_web_search else None
                    
                    # Handle web search if forced or enabled
                    context = ""
                    if st.session_state.force_web_search and web_search_agent:
                        with st.spinner("🔍 Searching the web..."):
                            try:
                                web_results = web_search_agent.run(prompt).content
                                if web_results:
                                    context = f"Web Search Results:\n{web_results}"
                                    st.info("ℹ️ Using web search as requested.")
                            except Exception as e:
                                st.error(f"❌ Web search error: {str(e)}")
                    
                    # Generate response
                    if context:
                        full_prompt = f"""Context: {context}

    Question: {prompt}

    Please provide a comprehensive answer based on the available information."""
                    else:
                        full_prompt = prompt

                    response = rag_agent.run(full_prompt)
                    response_content = response.content
                    
                    # Extract thinking process and final response
                    import re
                    think_pattern = r'<think>(.*?)</think>'
                    think_match = re.search(think_pattern, response_content, re.DOTALL)
                    
                    if think_match:
                        thinking_process = think_match.group(1).strip()
                        final_response = re.sub(think_pattern, '', response_content, flags=re.DOTALL).strip()
                    else:
                        thinking_process = None
                        final_response = response_content
                    
                    # Add assistant response to history (only the final response)
                    st.session_state.history.append({
                        "role": "assistant",
                        "content": final_response
                    })
                    
                    # Display assistant response
                    with st.chat_message("assistant"):
                        if thinking_process:
                            with st.expander("🤔 See thinking process"):
                                st.markdown(thinking_process)
                        st.markdown(final_response)

                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")

    else:
        st.warning("You can directly talk to r1 locally! Toggle the RAG mode to upload documents!")



if __name__ == "__main__":
    GlobalHydra.instance().clear() # Clear any previous configurations sinon il y'a erreur disant qu'il y'a déja une instance qui fonctionne.
    main()


#footer : 

#import tempfile
#from datetime import datetime
#from typing import List
#import bs4
#from agno.agent import Agent
#from agno.models.ollama import Ollama
#from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_qdrant import QdrantVectorStore
#from qdrant_client import QdrantClient
#from qdrant_client.models import Distance, VectorParams
#from langchain_core.embeddings import Embeddings
#from agno.tools.exa import ExaTools
#from agno.embedder.ollama import OllamaEmbedder