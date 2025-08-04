# Document Intelligence V4 - LLM-Powered RAG System

[![Build and deploy Python app to Azure Web App](https://github.com/Polymath-Saksh/document-intelligence-v4/actions/workflows/main_doc-qa-v4.yml/badge.svg)](https://github.com/Polymath-Saksh/document-intelligence-v4/actions/workflows/main_doc-qa-v4.yml)

## 📋 Overview

Document Intelligence V4 is an advanced Retrieval-Augmented Generation (RAG) system that leverages Azure AI services to answer domain-specific questions from large documents. The system is particularly designed for processing insurance, legal, HR, and compliance documents with high accuracy and semantic understanding.

## ✨ Key Features

- **🔍 Smart Document Processing**: Automated PDF download, text extraction, and intelligent chunking with overlap
- **🧠 Semantic Search**: Vector-based similarity search using Azure OpenAI embeddings (text-embedding-ada-002)
- **💬 Contextual Q&A**: GPT-4 powered question answering with document context
- **🔐 Secure API**: Bearer token authentication with FastAPI
- **⚡ High Performance**: Optimized chunking and batch embedding processing
- **🌐 Cloud-Ready**: Deployable on Azure Web Apps with CI/CD pipeline
- **📊 Comprehensive Logging**: Detailed performance monitoring and error tracking

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Document  │───▶│  Text Extraction │───▶│   Text Chunking │
│   (URL Input)   │    │   (PyPDF2)      │    │   (Overlap)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │◀───│  Answer Generation│◀───│   Embedding     │
│   Response      │    │    (GPT-4)       │    │   (ADA-002)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                       ┌──────────────────┐    ┌─────────────────┐
                       │  Semantic Search │◀───│  Vector Store   │
                       │   (Top-K)        │    │  (In-Memory)    │
                       └──────────────────┘    └─────────────────┘
```

### Technology Stack

- **Backend**: FastAPI with Python 3.12
- **Document Processing**: PyPDF2 for text extraction
- **AI Services**: Azure OpenAI (GPT-4, text-embedding-ada-002)
- **Vector Operations**: NumPy for similarity calculations
- **Authentication**: Bearer token security
- **Deployment**: Azure Web Apps
- **CI/CD**: GitHub Actions

## 🚀 Quick Start

### Prerequisites

- Python 3.12 or higher
- Azure OpenAI service with GPT-4 and embedding model deployments
- Azure Blob Storage (optional, for document storage)
- Azure AI Search (optional, for advanced indexing)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Polymath-Saksh/document-intelligence-v4.git
   cd document-intelligence-v4
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   
   Create a `.env` file in the root directory:
   ```bash
   # Azure OpenAI Configuration
   AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com/
   AZURE_OPENAI_API_KEY=your-api-key-here
   AZURE_OPENAI_API_VERSION=2024-12-01-preview
   AZURE_OPENAI_DEPLOYMENT=gpt-4  # Your GPT-4 deployment name
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002  # Your embedding deployment name

   # Security
   BEARER_TOKEN=your-secure-bearer-token

   # Optional: Azure Services
   AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
   AZURE_BLOB_CONNECTION_STRING=your-blob-connection-string
   AZURE_BLOB_CONTAINER=your-container-name
   ```

5. **Start the application**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

## 📖 API Documentation

### Authentication

All endpoints require Bearer token authentication. Include the token in the Authorization header:

```bash
Authorization: Bearer your-secure-bearer-token
```

### Main Endpoint

#### `POST /hackrx/run`

Process a PDF document and answer questions about its content.

**Request Body:**
```json
{
  "documents": "https://example.com/path/to/document.pdf",
  "questions": [
    "What is the coverage amount?",
    "Who is the policyholder?",
    "What are the exclusions?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "The coverage amount is $100,000 as stated in section 2.1.",
    "The policyholder is John Doe, policy number ABC123.",
    "Exclusions include natural disasters and pre-existing conditions."
  ]
}
```

**Example using cURL:**
```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Authorization: Bearer your-secure-bearer-token" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/sample.pdf",
    "questions": ["What is the main topic?"]
  }'
```

**Example using Python:**
```python
import requests

url = "http://localhost:8000/hackrx/run"
headers = {
    "Authorization": "Bearer your-secure-bearer-token",
    "Content-Type": "application/json"
}
data = {
    "documents": "https://example.com/sample.pdf",
    "questions": ["What is the main topic?"]
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
```

### Health Check Endpoints

- `GET /` - Welcome message
- `GET /test` - Health check endpoint

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI service endpoint | Yes | - |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | Yes | - |
| `AZURE_OPENAI_API_VERSION` | API version | No | `2024-12-01-preview` |
| `AZURE_OPENAI_DEPLOYMENT` | GPT-4 deployment name | Yes | - |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding model deployment | No | `text-embedding-ada-002` |
| `BEARER_TOKEN` | API authentication token | Yes | - |
| `AZURE_SEARCH_ENDPOINT` | Azure AI Search endpoint | No | - |
| `AZURE_BLOB_CONNECTION_STRING` | Blob storage connection | No | - |
| `AZURE_BLOB_CONTAINER` | Blob container name | No | - |

### Chunking Parameters

You can modify chunking behavior in `app/main.py`:

```python
# Adjust chunk size and overlap
chunks = chunk_text_overlap(text, chunk_size=500, overlap=100)

# Modify retrieval parameters
top_chunks = get_top_chunks(question, chunks, chunk_embeds, top_k=4)
```

## 🚀 Deployment

### Azure Web Apps

The repository includes GitHub Actions workflow for automatic deployment to Azure Web Apps.

1. **Configure Azure Web App**
   - Create an Azure Web App with Python 3.12 runtime
   - Configure the environment variables in Azure portal

2. **Set up GitHub Secrets**
   Add the following secrets to your GitHub repository:
   - `AZUREAPPSERVICE_CLIENTID_*`
   - `AZUREAPPSERVICE_TENANTID_*`
   - `AZUREAPPSERVICE_SUBSCRIPTIONID_*`

3. **Deploy**
   Push to `main` branch to trigger automatic deployment.

### Local Docker (Optional)

```bash
# Build Docker image
docker build -t document-intelligence-v4 .

# Run container
docker run -p 8000:8000 --env-file .env document-intelligence-v4
```

## 🔧 Development

### Project Structure

```
document-intelligence-v4/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py              # Configuration management
│   ├── openai_utils.py        # Azure OpenAI integration
│   ├── chunk_and_embed.py     # PDF processing and embedding
│   ├── azure_search_setup.py  # Azure AI Search configuration
│   ├── blob_utils.py          # Azure Blob Storage utilities
│   └── clause_logic.py        # Domain-specific processing logic
├── .github/workflows/         # CI/CD pipeline
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

### Adding New Features

1. **Custom Document Processing**: Extend `chunk_and_embed.py` for new document formats
2. **Enhanced Search**: Modify `azure_search_setup.py` for Azure AI Search integration
3. **Domain Logic**: Add business rules in `clause_logic.py`
4. **API Endpoints**: Extend `main.py` with new routes

### Running Tests

```bash
# Install development dependencies
pip install pytest pytest-asyncio httpx

# Run tests (if available)
python -m pytest

# Manual testing with API documentation
# Visit http://localhost:8000/docs
```

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Errors**
   ```
   Error: Invalid API key or endpoint
   ```
   - Verify `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`
   - Check deployment names match your Azure OpenAI resource

2. **PDF Download Failures**
   ```
   Error: PDF extraction failed
   ```
   - Ensure PDF URL is accessible and valid
   - Check network connectivity
   - Verify PDF is not password-protected

3. **Memory Issues with Large Documents**
   ```
   Error: Memory allocation failed
   ```
   - Reduce `chunk_size` parameter
   - Process documents in smaller batches
   - Consider using Azure AI Search for large-scale indexing

4. **Authentication Errors**
   ```
   Error: Invalid or missing token
   ```
   - Verify `BEARER_TOKEN` environment variable
   - Ensure correct Authorization header format

### Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Optimization

- **Batch Processing**: Process multiple questions efficiently
- **Caching**: Implement Redis for embedding caching
- **Async Processing**: Use background tasks for large documents
- **Azure AI Search**: Migrate to Azure AI Search for production scale

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** with appropriate tests
4. **Follow code style**: Use `black` for formatting, `flake8` for linting
5. **Commit your changes**: `git commit -m 'Add amazing feature'`
6. **Push to the branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Code Style

```bash
# Format code
black app/

# Lint code
flake8 app/

# Type checking
mypy app/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References

- [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Azure AI Search RAG Architecture](https://learn.microsoft.com/en-us/azure/search/retrieval-augmented-generation-overview)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Retrieval-Augmented Generation (RAG)](https://arxiv.org/abs/2005.11401)

## 📞 Support

For questions and support:
- Create an [issue](https://github.com/Polymath-Saksh/document-intelligence-v4/issues) on GitHub
- Review the [documentation](https://github.com/Polymath-Saksh/document-intelligence-v4/wiki)
- Check [discussions](https://github.com/Polymath-Saksh/document-intelligence-v4/discussions) for community help

---

**Built with ❤️ using Azure AI Services and FastAPI**
