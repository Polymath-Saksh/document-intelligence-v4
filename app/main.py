# type: ignore
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
import os
import time
import logging
from dotenv import load_dotenv
from app.chunk_and_embed import download_pdf
from app.file_utils import download_file, extract_text_from_file
from app.openai_utils import ask_llm, get_embedding
from pinecone import Pinecone 
from langchain.text_splitter import RecursiveCharacterTextSplitter 

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-app")

def chunk_text_overlap(text, chunk_size=1200, overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "]
    )
    return splitter.split_text(text)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "__default__")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_INDEX_HOST)


# Upsert all chunk embeddings to Pinecone and store chunk text as metadata (new SDK)
# Upsert all chunk texts to Pinecone (let Pinecone handle embedding)
import gc
def upsert_chunks_to_pinecone(chunks, batch_size=50):
    ids = []
    for batch_start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[batch_start:batch_start+batch_size]
        embeddings = get_embedding(batch_chunks)
        records = []
        for i, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
            chunk_id = f"chunk-{batch_start + i}"
            logger.info(f"Upserting chunk {batch_start + i}: {chunk[:120].replace('\n', ' ')} ...")
            records.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": {
                    "chunk_text": chunk,
                }
            })
            ids.append(chunk_id)
        index.upsert(
            vectors=records,
            namespace=PINECONE_NAMESPACE
        )
        del records, embeddings, batch_chunks
        gc.collect()
    return ids


# Query Pinecone for top-k similar chunks (new SDK, with integrated embedding)
def get_top_chunks(question, top_k=10):
    # Hybrid search: combine dense vector and keyword search
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
    # Keyword search (if supported by Pinecone index)
    try:
        keyword_results = index.query(
            namespace=PINECONE_NAMESPACE,
            filter={"chunk_text": {"$contains": question}},
            top_k=top_k,
            include_metadata=True
        )
    except Exception as e:
        logger.warning(f"Keyword search failed or not supported: {e}")
        keyword_results = {"matches": []}

    # Merge and deduplicate results (favoring dense first)
    seen = set()
    merged_chunks = []
    for match in dense_results.get('matches', []):
        chunk = match.get('metadata', {}).get('chunk_text', '')
        logger.info(f"Dense retrieved chunk: {chunk[:120].replace('\n', ' ')} ...")
        chunk_key = chunk[:50]
        if chunk_key not in seen:
            merged_chunks.append(chunk)
            seen.add(chunk_key)
        if len(merged_chunks) >= 4:
            break
    # Add keyword results if not already present
    for match in keyword_results.get('matches', []):
        chunk = match.get('metadata', {}).get('chunk_text', '')
        logger.info(f"Keyword retrieved chunk: {chunk[:120].replace('\n', ' ')} ...")
        chunk_key = chunk[:50]
        if chunk_key not in seen:
            merged_chunks.append(chunk)
            seen.add(chunk_key)
        if len(merged_chunks) >= 4:
            break
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

    # Step 1: Download and extract text from file (PDF, DOCX, EML)
    file_url = request.documents
    local_file = "temp_downloaded_file"
    t0 = time.time()
    try:
        logger.info(f"Downloading file from {file_url}")
        download_file(file_url, local_file)
        logger.info(f"Extracting text from {local_file}")
        text = extract_text_from_file(local_file)
        logger.info(f"Extracted {len(text)} characters from file")
    except Exception as e:
        logger.error(f"File extraction failed: {e}")
        raise HTTPException(status_code=400, detail=f"File extraction failed: {e}")
    t1 = time.time()
    logger.info(f"File download and extraction took {t1-t0:.2f} seconds")

    # Step 2: Chunk the text

    # Timing: Chunking
    t2 = time.time()
    logger.info("Chunking extracted text with overlap")
    chunks = chunk_text_overlap(text, chunk_size=500, overlap=100)
    logger.info(f"Generated {len(chunks)} overlapping chunks from document")
    t3 = time.time()
    logger.info(f"Chunking took {t3-t2:.2f} seconds")






    # Step 3: Upsert all chunk texts to Pinecone (embedding handled by Pinecone)
    t4 = time.time()
    logger.info("Upserting chunk texts to Pinecone index (embedding handled by Pinecone)")
    chunk_ids = upsert_chunks_to_pinecone(chunks)
    t5 = time.time()
    logger.info(f"Pinecone upsert took {t5-t4:.2f} seconds")

    # For each question, find relevant chunks and generate answer
    import asyncio

    # Limit concurrency for question processing
    semaphore = asyncio.Semaphore(10)  # Adjust concurrency as needed

    async def process_question(idx, question):
        async with semaphore:
            tq_start = time.time()
            logger.info(f"Processing question {idx+1}/{len(request.questions)}: {question}")
            # Run blocking get_top_chunks in a thread pool
            loop = asyncio.get_running_loop()
            top_chunks = await loop.run_in_executor(None, get_top_chunks, question, 8)
            logger.info(f"Selected top {len(top_chunks)} relevant chunks for question")
            context = "\n".join(top_chunks)
            concise_prompt = (
                "Answer concisely and only with facts from the context. "
                "If the answer is not present in the context, reply: 'Not found in document.'\n"
                f"Question: {question}\nContext: {context}"
            )
            try:
                # Run blocking ask_llm in a thread pool
                answer = await loop.run_in_executor(None, ask_llm, question, concise_prompt)
                logger.info(f"LLM answer generated successfully")
                # Fallback: if answer is empty or generic, try with more context
                if not answer or answer.strip().lower() in ["not found", "not found in document", "", "no answer"]:
                    logger.info("LLM returned empty or generic answer, retrying with more context chunks if available.")
                    more_chunks = await loop.run_in_executor(None, get_top_chunks, question, 16)
                    more_context = "\n".join(more_chunks)
                    retry_prompt = (
                        "Answer concisely and only with facts from the context. "
                        "If the answer is not present in the context, reply: 'Not found in document.'\n"
                        f"Question: {question}\nContext: {more_context}"
                    )
                    answer = await loop.run_in_executor(None, ask_llm, question, retry_prompt)
            except Exception as e:
                logger.error(f"Error generating answer: {e}")
                answer = f"Error generating answer: {e}"
            if not answer or answer.strip() == "":
                answer = "Not found in document."
            tq_end = time.time()
            logger.info(f"Total time for question {idx+1}: {tq_end-tq_start:.2f} seconds")
            return answer

    # Run all questions in parallel, but limit concurrency
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