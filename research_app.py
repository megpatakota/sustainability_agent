"""
Streamlit app for document research and analysis.

This app allows users to upload PDF files, process them with semantic search,
and interact with the documents through a chat interface.
"""

import asyncio
import hashlib
import os
import pickle
from io import BytesIO
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import faiss
import numpy as np
import streamlit as st
from fastembed import TextEmbedding
from agents import Agent, Runner, function_tool, trace
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Research Agent",
    page_icon="🔍",
    layout="centered"
)

# Cache directory for embeddings
CACHE_DIR = Path("data/embeddings_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def get_embedding_model():
    """Initialize and cache the embedding model."""
    return TextEmbedding()


def compute_file_hash(file_bytes: bytes) -> str:
    """Compute MD5 hash of file contents for caching."""
    return hashlib.md5(file_bytes).hexdigest()


def get_cache_path(file_hash: str) -> Path:
    """Get the cache path for a file hash."""
    return CACHE_DIR / f"{file_hash}.pkl"


def load_pdf_from_bytes(file_bytes: bytes) -> list[str]:
    """Load and extract text from PDF bytes."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = [page.get_text() for page in doc]
    doc.close()
    return pages


def load_or_create_embeddings(pages: list[str], file_hash: str, embedding_model: TextEmbedding) -> tuple[list, bool]:
    """
    Load embeddings from cache or create new ones.
    
    Returns:
        tuple: (embeddings_list, was_cached)
    """
    cache_path = get_cache_path(file_hash)
    
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            embeddings_list = pickle.load(f)
        return embeddings_list, True
    
    # Create new embeddings
    batch_size = 16
    embeddings_list = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_batches = (len(pages) + batch_size - 1) // batch_size
    
    for i, batch_start in enumerate(range(0, len(pages), batch_size)):
        batch = pages[batch_start : batch_start + batch_size]
        batch_embeddings = list(embedding_model.embed(batch))
        embeddings_list.extend(batch_embeddings)
        
        progress = (i + 1) / total_batches
        progress_bar.progress(progress)
        status_text.text(f"Processing batch {i + 1}/{total_batches}...")
    
    progress_bar.empty()
    status_text.empty()
    
    # Save to cache
    with open(cache_path, "wb") as f:
        pickle.dump(embeddings_list, f)
    
    return embeddings_list, False


def build_faiss_index(embeddings_list: list) -> faiss.Index:
    """Build FAISS index from embeddings."""
    embeddings_np = np.array(embeddings_list).astype("float32")
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    return index


def create_search_tool(pages: list[str], index: faiss.Index, embedding_model: TextEmbedding):
    """Create a search tool function for the agents."""
    @function_tool
    def search(query: str, top_k: int = 5) -> str:
        """
        Search the document using semantic similarity.

        Args:
            query: The search query text
            top_k: Number of top results to return (default: 5)

        Returns:
            A formatted string containing the top matching pages and their content
        """
        # Get embedding for the query text
        query_embedding = np.array(list(embedding_model.embed([query]))).astype("float32")

        # Search the FAISS index
        distances, indices = index.search(query_embedding, top_k)

        # Format the results
        results = []
        for match_idx, dist in zip(indices[0], distances[0]):
            page_content = pages[match_idx]
            results.append(
                f"--- Page {match_idx + 1} (distance {dist:.2f}) ---\n{page_content[:2000]}\n"
            )

        return "\n".join(results)
    
    return search


def create_agents(search_tool):
    """Create research and report agents."""
    # Research agent that uses the search tool to gather relevant information
    research_agent = Agent(
        name="research_agent",
        instructions="""You are a research agent that uses the search tool to find relevant information 
    from the document. When given a user query, you should:
    1. Break down the query into specific search terms
    2. Perform multiple searches to gather comprehensive information
    3. Synthesize the findings into a structured research summary incorporating verbatim quotes of the most relevant things you found, together with page number references for each quote.
    4. Ensure what you write can be used by someone else that needs to draft a research report based on your findings

    Use the search tool multiple times to explore different aspects of the query. Take your time, be persistent, be curious, follow leads, and do not stop searching until you have exhausted all relevant lines of inquiry.""",
        tools=[search_tool],
    )

    # Report drafting agent that creates a final report
    report_agent = Agent(
        name="report_agent",
        instructions="""You are a report drafting agent. Given research findings, you should:
    1. Organize the information into a clear, structured report
    2. Include relevant details and citations (page numbers)
    3. Write in a professional, comprehensive manner
    4. Ensure the report directly addresses the user's original query""",
        output_type=str,
    )
    
    return research_agent, report_agent


async def run_research_phase(user_query: str, research_agent: Agent):
    """Run the research phase of the workflow."""
    with trace("RAG Research Phase"):
        research_result = await Runner.run(
            research_agent,
            user_query,
        )
        return research_result.final_output


async def run_report_phase(user_query: str, research_findings: str, report_agent: Agent):
    """Run the report drafting phase of the workflow."""
    with trace("RAG Report Phase"):
        report_result = await Runner.run(
            report_agent,
            f"User Query: {user_query}\n\nResearch Findings:\n{research_findings}",
        )
        return report_result.final_output


def initialize_document(uploaded_file):
    """Initialize document processing and caching."""
    # Read file bytes
    file_bytes = uploaded_file.getvalue()
    file_hash = compute_file_hash(file_bytes)
    
    # Check if we already have this document loaded
    if "file_hash" in st.session_state and st.session_state.file_hash == file_hash:
        return True  # Already initialized
    
    # Process the file
    with st.spinner("Processing document..."):
        pages = load_pdf_from_bytes(file_bytes)
        
        # Get embedding model
        embedding_model = get_embedding_model()
        
        # Load or create embeddings
        embeddings_list, was_cached = load_or_create_embeddings(pages, file_hash, embedding_model)
        
        # Build FAISS index
        index = build_faiss_index(embeddings_list)
        
        # Create search tool
        search_tool = create_search_tool(pages, index, embedding_model)
        
        # Create agents
        research_agent, report_agent = create_agents(search_tool)
        
        # Store in session state
        st.session_state.pages = pages
        st.session_state.index = index
        st.session_state.embedding_model = embedding_model
        st.session_state.search_tool = search_tool
        st.session_state.research_agent = research_agent
        st.session_state.report_agent = report_agent
        st.session_state.file_hash = file_hash
        st.session_state.file_name = uploaded_file.name
    
    return True


def main():
    """Main Streamlit app."""
    st.title("🔍 Research Agent")
    st.caption("Upload a document and ask questions using AI-powered semantic search")
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Upload a PDF file to analyze. The document will be processed and cached for faster future access."
    )
    
    # Initialize document if uploaded
    if uploaded_file is not None:
        if initialize_document(uploaded_file):
            st.success(f"📄 **{st.session_state.file_name}** is ready for analysis!")
            st.info(f"📊 Document has {len(st.session_state.pages)} pages")
            
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": f"Hello! I've analyzed **{st.session_state.file_name}**. Ask me anything about the document, and I'll search through it and generate a comprehensive report for you."
                    }
                ]
            
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask a question about the document..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate response with separate spinner indicators
                try:
                    # Step 1: Research phase with spinner
                    with st.spinner("🔍 Searching document for relevant information..."):
                        research_findings = asyncio.run(
                            run_research_phase(prompt, st.session_state.research_agent)
                        )
                    
                    # Step 2: Report drafting phase with spinner
                    with st.spinner("📝 Drafting comprehensive report..."):
                        report = asyncio.run(
                            run_report_phase(prompt, research_findings, st.session_state.report_agent)
                        )
                    
                    # Display report as its own message
                    with st.chat_message("assistant"):
                        st.markdown(report)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": report})
                    
                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    with st.chat_message("assistant"):
                        st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        # Welcome message when no file is uploaded
        st.info("👆 Please upload a PDF file to get started")
        
        # Clear session state if file is removed
        if "file_hash" in st.session_state:
            for key in ["pages", "index", "embedding_model", "search_tool", 
                       "research_agent", "report_agent", "file_hash", "file_name", "messages"]:
                if key in st.session_state:
                    del st.session_state[key]


if __name__ == "__main__":
    main()

