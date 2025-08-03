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
