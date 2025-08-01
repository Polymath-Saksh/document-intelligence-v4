from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List

import os
import time
import logging
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-app")

# Import RAG pipeline components
from app.chunk_and_embed import download_pdf, extract_text_from_pdf
from app.openai_utils import ask_llm
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

def chunk_text_overlap(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(' '.join(chunk))
        i += chunk_size - overlap
    return chunks


# Load transformer once at startup
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def embed(texts):
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    token_embeddings = model_output[0]
    input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_top_chunks(question, chunks, chunk_embeds, top_k=1):
    question_embed = embed([question]).cpu().numpy()[0]
    similarities = np.dot(chunk_embeds, question_embed) / (np.linalg.norm(chunk_embeds, axis=1) * np.linalg.norm(question_embed) + 1e-9)
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]
app = FastAPI(title="Doc QA API - V4", description="API for document question answering using LLMs/embeddings.", root_path="/api/v1")
security = HTTPBearer()
BEARER_TOKEN = os.getenv("BEARER_TOKEN", "your-secure-token")

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")

@app.post("/hackrx/run/", response_model=QueryResponse)
async def run_query(request: QueryRequest, _: HTTPAuthorizationCredentials = Depends(verify_token)):
    # Step 1: Download and extract text from PDF
    pdf_url = request.documents
    local_pdf = "temp.pdf"


    # Timing: PDF download and extraction
    t0 = time.time()
    try:
        logger.info(f"Downloading PDF from {pdf_url}")
        download_pdf(pdf_url, local_pdf)
        logger.info(f"Extracting text from PDF {local_pdf}")
        text = extract_text_from_pdf(local_pdf)
        logger.info(f"Extracted {len(text)} characters from PDF")
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {e}")
    t1 = time.time()
    logger.info(f"PDF download and extraction took {t1-t0:.2f} seconds")

    # Step 2: Chunk the text


    # Timing: Chunking
    t2 = time.time()
    logger.info("Chunking extracted text with overlap")
    chunks = chunk_text_overlap(text, chunk_size=500, overlap=100)
    logger.info(f"Generated {len(chunks)} overlapping chunks from document")
    t3 = time.time()
    logger.info(f"Chunking took {t3-t2:.2f} seconds")


    # Step 3: Precompute chunk embeddings once

    # Timing: Embedding
    t4 = time.time()
    logger.info("Embedding all chunks once for semantic search")
    chunk_embeds = embed(chunks).cpu().numpy()
    t5 = time.time()
    logger.info(f"Chunk embedding took {t5-t4:.2f} seconds")

    # For each question, find relevant chunks and generate answer

    answers = []
    for idx, question in enumerate(request.questions):
        tq_start = time.time()
        logger.info(f"Processing question {idx+1}/{len(request.questions)}: {question}")
        top_chunks = get_top_chunks(question, chunks, chunk_embeds, top_k=4)
        logger.info(f"Selected top {len(top_chunks)} relevant chunks for question")
        context = "\n".join(top_chunks)
        try:
            logger.info(f"Calling LLM for answer generation")
            concise_prompt = f"Answer concisely and only with facts from the context.\nQuestion: {question}\nContext: {context}"
            answer = ask_llm(question, context=concise_prompt)
            logger.info(f"LLM answer generated successfully")
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            answer = f"Error generating answer: {e}"
        answers.append(answer)
        tq_end = time.time()
        logger.info(f"Total time for question {idx+1}: {tq_end-tq_start:.2f} seconds")

    logger.info(f"Returning {len(answers)} answers to client")
    return QueryResponse(answers=answers)
