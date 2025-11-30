# 🔍 Research Agent

An AI-powered document research and analysis tool that allows you to upload PDF documents and interact with them through a conversational interface. The system uses semantic search and AI agents to research documents and generate comprehensive reports.

## Problem Statement

Consulting workflows often require analyzing large, complex documents (such as Environmental Impact Assessment reports) that can span thousands of pages. Traditional keyword search and manual review are time-consuming and may miss relevant information scattered across different sections.

**The Challenge:**
- Consultants need to quickly find relevant information in large PDF documents
- Information may be spread across multiple sections and pages
- Manual review is slow and error-prone
- Synthesizing findings into reports requires significant time and effort

**The Solution:**
This Research Agent automates the document research and synthesis process by:
- Using semantic search to find relevant information based on meaning, not just keywords
- Leveraging AI agents to perform comprehensive research across the entire document
- Automatically generating structured reports with citations
- Providing an intuitive chat interface for natural language queries

This solution was developed as a proof-of-concept for automating consulting workflows, specifically targeting EIA document analysis where consultants need to quickly identify risks, mitigation strategies, and compliance requirements.

## Overview

This application provides a Streamlit-based web interface for document analysis. Upload a PDF, and the system will:
1. Extract and process the document text
2. Create semantic embeddings for efficient search
3. Build a FAISS vector index for fast similarity search
4. Use AI agents to research and answer questions about the document
5. Generate comprehensive reports with citations

## Features

- 📄 **PDF Document Processing**: Upload and process PDF files of any size
- 🔍 **Semantic Search**: Find relevant information using AI-powered semantic similarity
- 💾 **Intelligent Caching**: Automatically caches embeddings to speed up future processing
- 🤖 **AI Agent Workflow**: Two-stage agent system for research and report generation
- 💬 **Chat Interface**: Interactive chat interface for asking questions
- 📊 **Comprehensive Reports**: Generate detailed reports with page citations

## Architecture

The system follows a two-phase architecture:

### Document Ingestion Phase
1. **PDF Parser & Text Extraction**: Extracts raw text from uploaded PDFs
2. **Embedding Model**: Converts text into numerical embeddings using FastEmbed
3. **Caching System**: Checks for existing embeddings and saves new ones to disk
4. **FAISS Vector Index**: Builds a searchable vector index for fast similarity search

### Agent Workflow Phase
1. **Research Agent**: Uses semantic search to gather relevant information from the document
2. **Search Function Tool**: Performs semantic searches on the FAISS index
3. **Report Drafting Agent**: Synthesizes research findings into comprehensive reports
4. **Streamlit Interface**: Displays results to the user

### Architecture Diagram

![Architecture Diagram](architecture_diagram.png)

The diagram above illustrates the complete workflow from document upload through report generation, showing how the system processes documents, caches embeddings, and uses AI agents to research and synthesize information.

## Installation

### Prerequisites

- Python 3.10.16 or higher
- Poetry (for dependency management)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/megpatakota/sustainability_agent.git
   cd sustainability_agent
   ```

2. **Install dependencies using Poetry**:
   ```bash
   poetry install
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Run the application**:
   ```bash
   poetry run streamlit run research_app.py
   ```

   Or activate the Poetry shell first:
   ```bash
   poetry shell
   streamlit run research_app.py
   ```

## Configuration

### Streamlit Configuration

The app is configured via `.streamlit/config.toml`:

- **maxUploadSize**: Maximum file upload size (default: 1000MB, set to 0 for unlimited)
- **developmentMode**: Set to `false` for production

### Environment Variables

- `OPENAI_API_KEY`: Required for the AI agents to function

## Usage

1. **Start the application**:
   ```bash
   streamlit run research_app.py
   ```

2. **Upload a PDF document**:
   - Click "Upload a PDF document" in the web interface
   - Select your PDF file
   - Wait for processing to complete (first-time processing may take a few minutes)

3. **Ask questions**:
   - Once the document is processed, you'll see a chat interface
   - Type your question in the chat input
   - The system will:
     - Search the document for relevant information
     - Generate a comprehensive report
     - Display the report with citations

## How It Works

### Document Processing

1. When you upload a PDF, the system:
   - Computes an MD5 hash of the file contents
   - Checks if embeddings already exist in the cache
   - If cached: loads embeddings from disk (fast)
   - If not cached: processes the PDF, generates embeddings, and saves to cache

2. The document is split into pages, and each page is converted into a semantic embedding

3. A FAISS index is built for fast similarity search

### Query Processing

1. **Research Phase**: The Research Agent uses the search tool to find relevant information
   - Breaks down your query into search terms
   - Performs multiple semantic searches
   - Gathers comprehensive information with page references

2. **Report Generation Phase**: The Report Drafting Agent creates a structured report
   - Organizes findings into a clear structure
   - Includes citations and page numbers
   - Ensures the report addresses your query

## Project Structure

```
sustainability_agent/
├── research_app.py          # Main Streamlit application
├── pyproject.toml           # Poetry dependencies and project config
├── poetry.lock             # Locked dependency versions
├── .streamlit/
│   └── config.toml         # Streamlit configuration
├── data/
│   ├── embeddings_cache/   # Cached embeddings (by file hash)
│   └── ...                 # Other data files
├── architecture_diagram.png    # Architecture diagram
├── .env                    # Environment variables (not in git)
└── README.md               # This file
```

## Dependencies

Key dependencies include:

- **streamlit**: Web interface framework
- **openai-agents**: AI agent framework for research and report generation
- **fastembed**: Fast text embedding generation
- **faiss-cpu**: Vector similarity search
- **pymupdf**: PDF parsing and text extraction
- **numpy**: Numerical operations
- **python-dotenv**: Environment variable management

See `pyproject.toml` for the complete list of dependencies.

## Caching

The system implements intelligent caching:

- **Embedding Cache**: Embeddings are stored in `data/embeddings_cache/` using file hash as the filename
- **Automatic Detection**: Re-uploading the same file (even with a different name) will use cached embeddings
- **Cache Invalidation**: Changing the file contents will generate a new hash and create new embeddings

## Limitations

- Currently supports PDF files only
- Large documents may take time to process on first upload
- Requires an OpenAI API key for agent functionality
- Embeddings are stored locally and can consume disk space

## Troubleshooting

### File Upload Issues

If you encounter file size limits:
- Check `.streamlit/config.toml` and set `maxUploadSize = 0` for unlimited uploads

### API Key Issues

If agents fail to work:
- Ensure your `.env` file contains a valid `OPENAI_API_KEY`
- Check that the API key has sufficient credits

### Memory Issues

For very large documents:
- Consider processing on a machine with more RAM
- The embedding cache helps reduce processing time for repeated uploads

## Author

Meghana Patakota (megpatakota@gmail.com)