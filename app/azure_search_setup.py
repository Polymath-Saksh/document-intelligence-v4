from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, VectorSearch, VectorSearchAlgorithmConfiguration,
    SearchIndexerDataSourceConnection, SearchIndexerSkillset, SearchIndexer, InputFieldMappingEntry, OutputFieldMappingEntry
)
from azure.identity import DefaultAzureCredential
import os

# Set your Azure resource details

search_service_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT", "https://<your-search-service>.search.windows.net")
search_admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY", "<your-search-admin-key>")
blob_connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING", "<your-blob-connection-string>")
blob_container_name = os.getenv("AZURE_BLOB_CONTAINER", "<your-container-name>")

# 1. Create the index schema
def create_index():
    index_client = SearchIndexClient(endpoint=search_service_endpoint, credential=search_admin_key)
    fields = [
        SimpleField(name="id", type="Edm.String", key=True),
        SearchableField(name="content", type="Edm.String"),
        SimpleField(name="metadata", type="Edm.String"),
        SimpleField(name="embedding", type="Collection(Edm.Single)", searchable=True, vector_search_dimensions=1536)
    ]
    vector_search = VectorSearch(
        algorithm_configurations=[
            VectorSearchAlgorithmConfiguration(
                name="vector-config",
                kind="hnsw"
            )
        ]
    )
    index = SearchIndex(name="rag-index", fields=fields, vector_search=vector_search)
    index_client.create_index(index)

# 2. Create the data source (Blob Storage)
def create_data_source():
    index_client = SearchIndexClient(endpoint=search_service_endpoint, credential=search_admin_key)
    data_source = SearchIndexerDataSourceConnection(
        name="blob-datasource",
        type="azureblob",
        connection_string=blob_connection_string,
        container={"name": blob_container_name}
    )
    index_client.create_data_source_connection(data_source)

# 3. Create the skillset (Text Split + Embedding)
def create_skillset():
    index_client = SearchIndexClient(endpoint=search_service_endpoint, credential=search_admin_key)
    skillset = SearchIndexerSkillset(
        name="rag-skillset",
        skills=[
            # Add Text Split skill and Embedding skill here (see Azure docs for details)
        ]
    )
    index_client.create_skillset(skillset)

# 4. Create and run the indexer
def create_indexer():
    index_client = SearchIndexClient(endpoint=search_service_endpoint, credential=search_admin_key)
    indexer = SearchIndexer(
        name="rag-indexer",
        data_source_name="blob-datasource",
        target_index_name="rag-index",
        skillset_name="rag-skillset",
        field_mappings=[
            InputFieldMappingEntry(source_field_name="content", target_field_name="content"),
            OutputFieldMappingEntry(source_field_name="embedding", target_field_name="embedding")
        ]
    )
    index_client.create_indexer(indexer)
    index_client.run_indexer("rag-indexer")

if __name__ == "__main__":
    create_index()
    create_data_source()
    create_skillset()
    create_indexer()
