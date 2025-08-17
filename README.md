# LLM-Powered Intelligent Query–Retrieval System (RAG)

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=white" alt="Python"></a>
<a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-005571?logo=fastapi&logoColor=white" alt="FastAPI"></a>
<a href="https://azure.microsoft.com/en-us/products/ai-services/openai-service/"><img src="https://img.shields.io/badge/Azure%20OpenAI-0078D4?logo=microsoftazure&logoColor=white" alt="Azure OpenAI"></a>
<a href="https://www.pinecone.io/"><img src="https://img.shields.io/badge/Pinecone-45B8AC?logo=pinecone&logoColor=white" alt="Pinecone"></a>
<a href="https://azure.microsoft.com/en-us/products/storage/blobs/"><img src="https://img.shields.io/badge/Azure%20Blob%20Storage-0089D6?logo=microsoftazure&logoColor=white" alt="Azure Blob Storage"></a>
<a href="https://pydantic-docs.helpmanual.io/"><img src="https://img.shields.io/badge/Pydantic-008000?logo=pydantic&logoColor=white" alt="Pydantic"></a>
<a href="https://www.uvicorn.org/"><img src="https://img.shields.io/badge/Uvicorn-22C55E?logo=uvicorn&logoColor=white" alt="Uvicorn"></a>

<a href="https://creativecommons.org/licenses/by-nc-sa/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg?logo=creativecommons&logoColor=white" alt="License: CC BY-NC-SA 4.0"></a>

## Overview

This project is an intelligent Retrieval-Augmented Generation (RAG) system for answering domain-specific questions from large documents (insurance, legal, HR, compliance, etc.). It leverages Azure OpenAI, Pinecone, and FastAPI to provide accurate, context-aware answers using LLMs and vector search.

## Architecture

- **Document Ingestion:** Documents are uploaded to Azure Blob Storage.
- **Indexing & Chunking:** Documents are chunked and embedded using Azure OpenAI, then indexed in Pinecone for fast vector retrieval.
- **Vector Database:** Pinecone stores vectorized document chunks for semantic search.
- **LLM Integration:** Azure OpenAI (GPT-4) is used for answer generation, grounded in retrieved document context.
- **API Layer:** FastAPI serves as the backend, exposing endpoints for document Q&A.
- **Deployment:** Designed for Azure Web Apps, but can run locally.

## Technologies Used

- **Python 3.12+**
- **FastAPI** – API framework
- **Azure OpenAI** – Embeddings and LLM completions
- **Pinecone** – Vector database for retrieval
- **Azure Blob Storage** – Document storage
- **PyPDF2, python-docx** – Document parsing
- **LangChain** – Text splitting
- **Uvicorn** – ASGI server
- **Requests, httpx** – HTTP clients
- **Pydantic** – Data validation

## Setup

1. **Configure Azure and Pinecone credentials** in `app/config.py` or via environment variables.
2. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

3. **Run the FastAPI app:**

   ```sh
   uvicorn app.main:app --reload
   ```

## Usage

POST to `/hackrx/run` with a Bearer token:

```json
{
	"documents": "<PDF URL>",
	"questions": ["Question 1", "Question 2"]
}
```

Sample `curl` request:

```sh
curl -X POST "http://localhost:8000/hackrx/run" \
   -H "Authorization: Bearer <YOUR_TOKEN_HERE>" \
   -H "Content-Type: application/json" \
   -d '{
      "documents": "https://example.com/sample.pdf",
      "questions": ["What is the policy number?", "Who is the contact person?"]
   }'
```

Response:

```json
{ "answers": ["...", "..."] }
```

## Customization

- Add advanced clause matching in `app/clause_logic.py`
- Integrate more business logic in `app/main.py` for your domain

## Contributors

- [**Saksham Kumar**](https://github.com/Polymath-Saksh) (Project Author & Maintainer)
- [**Aloukik Joshi**](https://github.com/aloukikjoshi) (Project Collaborator)
- [**Nihal Pandey**](https://github.com/NihalPandey5060) (Project Collaborator)

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](LICENSE).

## References

- [Azure RAG Architecture](https://learn.microsoft.com/en-us/azure/search/?wt.mc_id=studentamb_217334retrieval-augmented-generation-overview)
- [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/?wt.mc_id=studentamb_217334)
- [FastAPI](https://fastapi.tiangolo.com/)
