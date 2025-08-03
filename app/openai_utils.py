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
            {"role": "system", "content": "You are a helpful assistant that provides accurate and factual answers based on the provided document. If a question contains multiple parts, answer each part separately. Keep your answers concise and to the point, limiting them to one or two sentences per part. If the answer is not present in the context, respond with 'Not found in document.'"},
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