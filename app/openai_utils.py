import os
from openai import AzureOpenAI #type: ignore

def get_openai_client():
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "<your-openai-api-key>")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    return AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint
    )

def ask_llm(question: str, context: str) -> str:
    client = get_openai_client()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an assistant which helps users find information from documents. Keep the answers limited to 1-2 sentences, only based on the provided context."},
            {"role": "user", "content": f"{question}\nContext: {context}"}
        ],
        max_completion_tokens=800,
        temperature=1.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        model=deployment
    )
    return response.choices[0].message.content

def get_embedding(text: str, model: str = "text-embedding-ada-002") -> list:
    client = get_openai_client()
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding
