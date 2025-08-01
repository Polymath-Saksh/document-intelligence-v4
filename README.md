# LLM-Powered Intelligent Query–Retrieval System (RAG)

## Overview

This app uses Azure AI Search, Azure OpenAI, and FastAPI to answer domain-specific questions from large documents (insurance, legal, HR, compliance).

## Architecture

- Input Doc: Azure Blob Storage
- Indexer: Azure AI Search (chunking + vectorization)
- Vector DB: Azure AI Search
- LLM: Azure OpenAI (GPT-4)
- API: FastAPI on Azure Web Apps

## Setup

1. Fill in your Azure resource details in `config.py` or set environment variables.
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run Azure Search setup:
   ```sh
   python app/azure_search_setup.py
   ```
4. Start FastAPI app:
   ```sh
   uvicorn app.main:app --reload
   ```

## Usage

POST to `/hackrx/run` with Bearer token:

```
{
  "documents": "<PDF URL>",
  "questions": ["Question 1", "Question 2"]
}
```

Response:

```
{"answers": ["...", "..."]}
```

## Customization

- Add advanced clause matching in `clause_logic.py`
- Integrate more logic in `main.py` for domain-specific scenarios

## References

- [Azure RAG Architecture](https://learn.microsoft.com/en-us/azure/search/retrieval-augmented-generation-overview)
- [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [FastAPI](https://fastapi.tiangolo.com/)
