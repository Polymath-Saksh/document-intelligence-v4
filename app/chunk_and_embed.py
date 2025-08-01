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
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

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
