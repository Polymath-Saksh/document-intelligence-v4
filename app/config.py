import os

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "https://<your-search-service>.search.windows.net")
AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING", "<your-blob-connection-string>")
AZURE_BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER", "<your-container-name>")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://<your-openai-endpoint>")
BEARER_TOKEN = os.getenv("BEARER_TOKEN", "your-secure-token")
