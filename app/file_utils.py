import os
import mimetypes
import requests
import httpx
import asyncio
import PyPDF2
from docx import Document
import tempfile
from email import policy
from email.parser import BytesParser

# Download any file

def download_file(url, filename):
    """Synchronous file download (legacy)"""
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)

async def async_download_file(url, filename):
    """Async file download using httpx.AsyncClient"""
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            with open(filename, 'wb') as f:
                async for chunk in response.aiter_bytes():
                    f.write(chunk)

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# Extract text from DOCX
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return "\n".join([p.text for p in doc.paragraphs])

# Extract text from EML (email)
def extract_text_from_eml(eml_path):
    with open(eml_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    parts = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                parts.append(part.get_content())
    else:
        parts.append(msg.get_content())
    return "\n".join(parts)

# Main function to extract text based on file type
def extract_text_from_file(file_path):
    mime, _ = mimetypes.guess_type(file_path)
    # Try by MIME type first
    try:
        if mime == 'application/pdf' or file_path.lower().endswith('.pdf'):
            return extract_text_from_pdf(file_path)
        elif mime == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' or file_path.lower().endswith('.docx'):
            return extract_text_from_docx(file_path)
        elif mime == 'message/rfc822' or file_path.lower().endswith('.eml'):
            return extract_text_from_eml(file_path)
    except Exception as e:
        pass  # Fallback to trying all formats below

    # Fallback: try all supported formats in order
    errors = []
    for extractor, desc in [
        (extract_text_from_pdf, 'PDF'),
        (extract_text_from_docx, 'DOCX'),
        (extract_text_from_eml, 'EML'),
    ]:
        try:
            return extractor(file_path)
        except Exception as e:
            errors.append(f"{desc} extraction failed: {e}")
    raise ValueError(f"Unsupported file type: {mime} for {file_path}. Tried all extractors. Errors: {' | '.join(errors)}")

# app/openai_utils.py
import os
from openai import AzureOpenAI #type: ignore

def get_openai_client():
    """Initializes and returns an Azure OpenAI client."""
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "<your-openai-api-key>")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    return AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint
    )

def ask_llm(prompt: str) -> str:
    """
    Sends a pre-formatted prompt to the LLM and returns a concise, factual answer.
    """
    client = get_openai_client()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-nano")
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides accurate and factual answers based on the provided document. If a question contains multiple parts, answer each part separately. If the answer is not present in the context, respond with 'Not found in document.'"},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=800,
        temperature=1.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        model=deployment
    )
    return response.choices[0].message.content

def get_embedding(texts, model: str = "text-embedding-ada-002") -> list:
    """
    Accepts a string or a list of strings. Returns a list of embeddings (one per input).
    """
    client = get_openai_client()
    if isinstance(texts, str):
        texts = [texts]
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    return [item.embedding for item in response.data]