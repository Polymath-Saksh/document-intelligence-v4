# type: ignore
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
import os
import time
import logging
from dotenv import load_dotenv
from app.file_utils import async_download_file, extract_text_from_file
from app.openai_utils import ask_llm, get_embedding
from app.contact_utils import is_contact_question, extract_contact_details
from pinecone import Pinecone 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import gc
import asyncio
import urllib.parse

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-app")

# Pinecone setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "__default__")

# Initialize Pinecone and index
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_INDEX_HOST)

def chunk_text_overlap(text, chunk_size=1200, overlap=200):
    """
    Splits text into chunks with a specified overlap for better context.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "]
    )
    return splitter.split_text(text)

def upsert_chunks_to_pinecone(chunks, batch_size=100):
    """
    Upsert chunks to Pinecone in batches. This is a simplified version.
    """
    chunk_ids = []
    # In a real-world app, you'd generate embeddings for the chunks here.
    # For this example, let's assume we have a way to do that.
    # The original file used get_embedding, which we'll keep.
    try:
        embeddings = get_embedding(chunks)
        records = [{
            "id": f"chunk-{i}", 
            "values": embedding,
            "metadata": {"chunk_text": chunk}
        } for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))]
        
        index.upsert(vectors=records, namespace=PINECONE_NAMESPACE)
        chunk_ids = [record['id'] for record in records]
        
    except Exception as e:
        logger.error(f"Pinecone upsert failed: {e}")

    return chunk_ids

def get_top_chunks(question, top_k=20):
    """
    Queries Pinecone for top-k similar chunks using dense vector search.
    """
    query_embedding = get_embedding(question)
    if isinstance(query_embedding, list) and len(query_embedding) == 1:
        query_embedding = query_embedding[0]
    
    # Dense vector search
    dense_results = index.query(
        namespace=PINECONE_NAMESPACE,
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )
    
    merged_chunks = [match.get('metadata', {}).get('chunk_text', '') 
                     for match in dense_results.get('matches', [])]
    return merged_chunks

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
    
@app.post("/hackrx/run", response_model=QueryResponse)
@app.post("/hackrx/run/", response_model=QueryResponse)
async def run_query(request: QueryRequest, background_tasks: BackgroundTasks, _: HTTPAuthorizationCredentials = Depends(verify_token)):

    # Step 1: Download and extract text from file
    file_url = request.documents
    parsed_url = urllib.parse.urlparse(file_url)
    file_name_from_url = os.path.basename(parsed_url.path)
    _, ext = os.path.splitext(file_name_from_url)
    if not ext:
        ext = '.pdf' # Assume PDF if no extension is found
    local_file = f"temp_downloaded_file{ext}"
    t0 = time.time()
    try:
        logger.info(f"Downloading file from {file_url} (async)")
        await async_download_file(file_url, local_file)
        logger.info(f"Extracting text from {local_file}")
        text = extract_text_from_file(local_file)
        logger.info(f"Extracted {len(text)} characters from file")
    except Exception as e:
        logger.error(f"File extraction failed: {e}")
        raise HTTPException(status_code=400, detail=f"File extraction failed: {e}")
    t1 = time.time()
    logger.info(f"File download and extraction took {t1-t0:.2f} seconds")
    
    # Extract all contact details from the full document once
    all_contact_info = extract_contact_details(text)
    all_contact_hint = ""
    if any(all_contact_info.values()):
        all_contact_hint = "\nEmails: " + ", ".join(all_contact_info["emails"]) if all_contact_info["emails"] else ""
        all_contact_hint += "\nToll-free: " + ", ".join(all_contact_info["phones"]) if all_contact_info["phones"] else ""
        all_contact_hint += "\nAddresses: " + ", ".join(all_contact_info["addresses"]) if all_contact_info["addresses"] else ""
    
    # Step 2: Chunk the text and upsert to Pinecone
    t2 = time.time()
    logger.info("Chunking extracted text with overlap")
    chunks = chunk_text_overlap(text, chunk_size=500, overlap=100)
    logger.info(f"Generated {len(chunks)} overlapping chunks from document")
    t3 = time.time()
    logger.info(f"Chunking took {t3-t2:.2f} seconds")

    t4 = time.time()
    logger.info("Upserting chunk texts to Pinecone index")
    chunk_ids = upsert_chunks_to_pinecone(chunks)
    t5 = time.time()
    logger.info(f"Pinecone upsert took {t5-t4:.2f} seconds")
    
    # Semaphore to limit concurrency for LLM calls
    semaphore = asyncio.Semaphore(5)

    async def process_question(idx, question):
        async with semaphore:
            tq_start = time.time()
            logger.info(f"Processing question {idx+1}/{len(request.questions)}: {question}")
            
            # Normal chunk retrieval and LLM for all questions.
            loop = asyncio.get_running_loop()
            top_chunks = await loop.run_in_executor(None, get_top_chunks, question, 20)
            logger.info(f"Selected top {len(top_chunks)} relevant chunks for question")
            
            prompt_context_parts = []
            if all_contact_hint:
                prompt_context_parts.append(f"CONTACTS AND ADDRESSES IN THE DOCUMENT: {all_contact_hint}")
            
            prompt_context_parts.append("\n\nRELEVANT DOCUMENT CHUNKS:\n" + "\n".join(top_chunks))
            
            final_context = "\n".join(prompt_context_parts)
            
            prompt = (
                f"Question: {question}\nContext: {final_context}"
            )
            
            try:
                answer = await loop.run_in_executor(None, ask_llm, prompt)
                logger.info("LLM answer generated successfully")
            except Exception as e:
                logger.error(f"Error generating answer: {e}")
                answer = f"Error generating answer: {e}"
            
            if not answer or answer.strip() == "":
                answer = "Not found in document."
            tq_end = time.time()
            logger.info(f"Total time for question {idx+1}: {tq_end-tq_start:.2f} seconds")
            return answer

    # Run all questions in parallel
    answers = await asyncio.gather(*[process_question(idx, q) for idx, q in enumerate(request.questions)])
    logger.info(f"Returning {len(answers)} answers to client")
    
    # Cleanup: delete the upserted chunks from Pinecone in the background
    def cleanup_chunks(chunk_ids):
        try:
            if chunk_ids:
                logger.info(f"Cleaning up {len(chunk_ids)} chunks from Pinecone index.")
                index.delete(ids=chunk_ids, namespace=PINECONE_NAMESPACE)
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

    background_tasks.add_task(cleanup_chunks, chunk_ids)
    
    return QueryResponse(answers=answers)

# Root route for homepage
@app.get("/")
def homepage():
    return {"message": "Welcome to Doc QA API. Visit /docs for API documentation."}

# Keep /test for health check
@app.get("/test")
def root():
    return {"message": "Doc QA API is running."}
