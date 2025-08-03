import pinecone
def get_top_chunks(query, top_n=8):
    """
    Embed the query, retrieve top N relevant chunks from Pinecone, and return their texts.
    Assumes Pinecone index stores chunk text in metadata['text'].
    """
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "<your-openai-api-key>")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai_client = AzureOpenAI(
        api_key=api_key,
        api_version="2024-06-01",
        azure_endpoint=endpoint
    )
    # Embed the query
    query_embedding = openai_client.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    ).data[0].embedding

    # Pinecone setup
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
    pinecone_index = os.getenv("PINECONE_INDEX_HOST")
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
    index = pinecone.Index(pinecone_index)
    results = index.query(vector=query_embedding, top_k=top_n, include_metadata=True)
    top_chunks = [match['metadata']['text'] for match in results['matches']]
    return top_chunks

def build_context_for_prompt(top_chunks):
    """
    Concatenate top chunks for use as context in the prompt.
    """
    return "\n\n".join(top_chunks)

def build_prompt(context, question):
    """
    Build a prompt for GPT-4.1-nano using the provided context and question.
    """
    return f"""Answer the following question using only the provided context. If the answer is not present, say 'Not found in document.'\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"""
import PyPDF2
import requests
from openai import AzureOpenAI
import os

def download_pdf(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=1000):
    import re
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    current_length = 0
    overlap = 3  # Number of sentences to overlap between chunks
    for sentence in sentences:
        if current_length + len(sentence) > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            # Start new chunk with overlap
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_length = sum(len(s) for s in current_chunk)
        current_chunk.append(sentence)
        current_length += len(sentence)
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def get_embeddings(chunks):
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "<your-openai-api-key>")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai_client = AzureOpenAI(
        api_key=api_key,
        api_version="2024-06-01",
        azure_endpoint=endpoint
    )
    embeddings = []
    for chunk in chunks:
        response = openai_client.embeddings.create(
            input=chunk,
            model="text-embedding-ada-002"
        )
        embeddings.append(response.data[0].embedding)
    return embeddings

if __name__ == "__main__":
    url = os.getenv("PDF_URL")
    filename = "temp.pdf"
    download_pdf(url, filename)
    text = extract_text_from_pdf(filename)
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)
    print(f"Extracted {len(chunks)} chunks and embeddings.")

    # Example QA usage:
    user_query = "<your-question-here>"
    top_chunks = get_top_chunks(user_query, top_n=8)
    context = build_context_for_prompt(top_chunks)
    prompt = build_prompt(context, user_query)
    print("Prompt for GPT-4.1-nano:\n", prompt)
