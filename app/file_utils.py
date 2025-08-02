import os
import mimetypes
import requests
import PyPDF2
from docx import Document
import tempfile
from email import policy
from email.parser import BytesParser

# Download any file
def download_file(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)

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
    if mime == 'application/pdf':
        return extract_text_from_pdf(file_path)
    elif mime == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        return extract_text_from_docx(file_path)
    elif mime == 'message/rfc822' or file_path.lower().endswith('.eml'):
        return extract_text_from_eml(file_path)
    else:
        raise ValueError(f"Unsupported file type: {mime} for {file_path}")
