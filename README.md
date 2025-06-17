# Multi-Agent RAG with MLOPS infrastructure

[![Deepseek Agent Logo]([https://example.com/deepseek-logo.png](https://drive.google.com/file/d/1tFSWXyk1npec7np0vMjjJ5fUkWqT1eJF/view?usp=sharing))]([https://example.com](https://drive.google.com/file/d/1tFSWXyk1npec7np0vMjjJ5fUkWqT1eJF/view?usp=sharing))

A powerful Multi-agent RAG that combines local **Deepseek** models with **RAG** capabilities and integrates multiple models, including **Qwen2.5:4b**, **Llama3.2**, and **Deepseek**. Built using **Deepseek** (via Ollama), **Snowflake** for embeddings, **Qdrant** for vector storage, **Langtrace** for observability, and **Hydra** for configuration management, this application offers both simple local chat and advanced RAG-enhanced interactions with comprehensive document processing, web search capabilities, and an end-to-end **MLOps** pipeline for streamlined deployment and management.

## Features

- **Dual Operation Modes**
  - **Local Chat Mode**: Direct interaction with **Deepseek** locally
  - **RAG Mode**: Enhanced reasoning with document context and web search integration (llama3.2)

- **Multi-Agent System**: Combines various models, including **Qwen2.5:4b**, **Llama3.2**, and **Deepseek**, to enhance reasoning and improve the accuracy of responses.
  
- **Document Processing** (RAG Mode)
  - PDF document upload and processing
  - Web page content extraction
  - Automatic text chunking and embedding
  - Vector storage in **Qdrant** cloud

- **Intelligent Querying** (RAG Mode)
  - RAG-based document retrieval
  - Similarity search with threshold filtering
  - Automatic fallback to web search
  - Source attribution for answers

- **Advanced Capabilities**
  - **Exa AI** web search integration
  - Custom domain filtering for web search
  - Context-aware response generation
  - Chat history management
  - Thinking process visualization

- **MLOps Pipeline**
  - **Code Versioning with GitHub**: Managing code changes and collaboration
  - **Data Versioning with DVC**: Tracking changes in large datasets
  - **Continuous Integration with CML**: Automating ML workflows and testing
  - **Secrets Management**: Securing sensitive information
  - **Configuration Management with Hydra**: Organizing and managing configurations
  - **Observability using Langtrace**
  - **RAG performance tracking using RAGAS**

- **Model Specific Features**
  - **Flexible model selection**:
    - **Deepseek r1 1.5b** (lighter, suitable for most laptops)
    - **Deepseek r1 7b** (more capable, requires better hardware)
    - **Qwen2.5:4b** (advanced capabilities for more complex tasks)
    - **Llama3.2** (latest model for enhanced reasoning)
  - **Snowflake Arctic Embedding model** (SOTA) for vector embeddings
  - **Agno Agent framework** for orchestration
  - **Streamlit-based interactive interface**

## Prerequisites

### 1. Ollama Setup
1. Install [Ollama](https://ollama.ai)
2. Pull the **Deepseek** r1 model(s):
```bash
# For the lighter model
ollama pull deepseek-r1:1.5b

# For the more capable model (if your hardware supports it)
ollama pull deepseek-r1:7b

ollama pull snowflake-arctic-embed
ollama pull llama3.2
ollama pull qwen2.5:4b
````

### 2. Qdrant Cloud Setup (for RAG Mode)

1. Visit [Qdrant Cloud](https://cloud.qdrant.io/)
2. Create an account or sign in
3. Create a new cluster
4. Get your credentials:

   * Qdrant API Key: Found in API Keys section
   * Qdrant URL: Your cluster URL (format: `https://xxx-xxx.cloud.qdrant.io`)

### 3. Exa AI API Key (Optional)

1. Visit [Exa AI](https://exa.ai)
2. Sign up for an account
3. Generate an API key for web search capabilities

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git
cd rag_tutorials/deepseek_local_rag_agent
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run deepseek_rag_agent.py
```

## How the Application Works

The **Deepseek Local RAG Reasoning Agent** uses multiple components to provide advanced reasoning capabilities in **RAG Mode**. Here's how it works:

1. **Model Selection**: You can select between **Deepseek r1 1.5b**, **Deepseek r1 7b**, **Qwen2.5:4b**, and **Llama3.2**, depending on your hardware capability and task complexity.
2. **Multi-Agent System**: Multiple models interact to provide more accurate and flexible reasoning capabilities. This system combines models like **Qwen2.5:4b**, **Llama3.2**, and **Deepseek**.
3. **Document Upload**: In **RAG Mode**, you can upload PDF documents or input URLs to process content for knowledge retrieval. This data is embedded and stored in **Qdrant**.
4. **Querying**: The application enables intelligent querying with the **RAG** approach to search documents or fallback to web searches using **Exa AI**.
5. **Web Search Fallback**: When relevant documents aren't found, the system automatically switches to web search, using customizable domains (e.g., arxiv.org, wikipedia.org).

The app can be easily managed via the Streamlit interface, allowing toggling between different modes and configurations.

### Configuration

* **RAG Mode**: Enable or disable **RAG Mode** from the sidebar to activate document-based reasoning.
* **Web Search Fallback**: You can toggle web search in case documents aren’t available in the vector database.
* **Model Version**: Choose from **Deepseek r1 1.5b**, **Deepseek r1 7b**, **Qwen2.5:4b**, or **Llama3.2** models based on your hardware.

---

## Main Script

The core functionality is in the `deepseek_rag_agent.py` script, which handles the application’s logic. Here's an overview of the main components:

1. **Langtrace** for tracking requests and API calls.
2. **OllamaEmbedder**: Embeds documents for RAG-based retrieval.
3. **Qdrant** for vector storage and search capabilities.
4. **Agno** for agent orchestration.
5. **Streamlit**: Provides an interactive web interface for the user to interact with the agent.

The script initializes models, processes inputs (PDFs or URLs), and uses **Qdrant** to store vectors for document search and retrieval.

---

