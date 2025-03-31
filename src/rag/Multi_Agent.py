
"This file define the multiagent architecture of our RAG"

############ Always import this two libraries from Langtrace first of all #######################

from langtrace_python_sdk.utils.with_root_span import with_langtrace_root_span

###################################################################################################



import streamlit as st
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.exa import ExaTools



@with_langtrace_root_span()
def get_web_search_agent() -> Agent:
    """Initialize a web search agent."""
    return Agent(
        name="Web Search Agent",
        model=Ollama(id="llama3.2"),
        tools=[ExaTools(
            api_key=st.session_state.exa_api_key,
            include_domains=search_domains,
            num_results=5
        )],
        instructions="""You are a web search expert. Your task is to:
        1. Search the web for relevant information about the query
        2. Compile and summarize the most relevant information
        3. Include sources in your response
        """,
        show_tool_calls=True,
        markdown=True,
    )

@with_langtrace_root_span()
def get_rag_agent() -> Agent:
    """Initialize the main RAG agent."""
    return Agent(
        name="DeepSeek RAG Agent",
        model=Ollama(id=st.session_state.model_version),
        instructions="""You are an Intelligent Agent specializing in providing accurate answers.

        When asked a question:
        - Analyze the question and answer the question with what you know.
        
        When given context from documents:
        - Focus on information from the provided documents
        - Be precise and cite specific details
        
        When given web search results:
        - Clearly indicate that the information comes from web search
        - Synthesize the information clearly
        
        Always maintain high accuracy and clarity in your responses.
        """,
        show_tool_calls=True,
        markdown=True,
    )