##################################################################################################################################################
import requests
from dotenv import load_dotenv
import os
import json
from typing import List, Dict, Any, Optional, Union
import base64

prompt_template = f"""
You are an assistant for the 'Tools in Data Science' course. Given a question and context, respond with:

- A concise, accurate answer in plain English (no markdown or formatting).
- in JSON format.
"""

tools = [{
    "type": "function",
    "function": {
        "name": "generate_structured_response",
        "description": "Generate a structured response with an answer and URLs of the contexts used",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The answer to the user's question"
                },
                "links": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string", 
                                "description": "URL of the context used"
                            },
                            "text": {
                                "type": "string",
                                "description": "Very brief justification of why this source is relevant to the answer"
                            }
                        }
                    },
                    "description": "Relevant sources that support the answer"
                }
            },
            "required": ["answer"]
        }
    }
}]

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "mistralai/mistral-small-3.1-24b-instruct:free"
OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
    "Content-Type": "application/json"
}

def get_image_description(image: str) -> str:
    """
    Get a description of the image using the OpenRouter API.

    Args:
        image (str): Base64-encoded image string.

    Returns:
        str: The description of the image.
    """
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Please provide a detailed description of what this image contains."},{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}]}
        ]
    }
    response = requests.post(OPENROUTER_URL, headers=OPENROUTER_HEADERS, json=payload)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

def generate_answer(question: str, context: str) -> str:
    """
    Generate an answer to a question using the OpenRouter API.

    Args:
        question (str): The question to answer.
        context (str): Context to help answer the question.

    Returns:
        str: The generated answer.
    """
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": [{"type": "text", "text": question}]},
            {"role": "assistant", "content": [{"type": "text", "text": context}]}
        ],
        "tools": tools,
        "tool_choice": "required"
    }
    response = requests.post(OPENROUTER_URL, headers=OPENROUTER_HEADERS, json=payload)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['tool_calls'][0]['function']['arguments']

def get_embedding(text: str) -> List[float]:
    """
    Get the embedding for a given text using the OpenRouter API.

    Args:
        text (str): The text to embed.

    Returns:
        List[float]: The embedding vector.
    """
    nomic_url = "https://api-atlas.nomic.ai/v1/embedding/text"
    nomic_model = "nomic-embed-text-v1.5"
    nomic_headers = {
        "Authorization": f"Bearer {os.environ.get('NOMIC_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": nomic_model,
        "task_type": "search_document",
        "texts": [text]
    }
    response = requests.post(nomic_url, headers=nomic_headers, json=payload)
    response.raise_for_status()
    return response.json()['embeddings'][0]
    
    
######################################################################################################################################################




from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import numpy as np
import os
import base64
from llms import Nomic, OpenRouter, Ollama
import json

print("DEBUG: Initializing FastAPI app...")
app = FastAPI()
print("DEBUG: Creating Nomic instance...")
nomic = Nomic()
print("DEBUG: Creating OpenRouter client...")
client = OpenRouter()
print("DEBUG: App initialization complete.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class QuestionRequest(BaseModel):
    question: str
    image: str = None

def load_embeddings():
    """Load embeddings from a file or database."""
    print("DEBUG: Loading embeddings from tds_course.npz...")
    data = np.load("tds_course.npz", allow_pickle=True)
    chunks = data["chunks"]
    embeddings = data["embeddings"]
    print(f"DEBUG: Loaded {len(chunks)} chunks and {len(embeddings)} embeddings")
    print(f"DEBUG: Embedding shape: {embeddings.shape}")
    return chunks, embeddings

def answer_question(question: str, image: str = None) -> Dict[str, str]:
    """
    Process the question and image to generate an answer.
    
    Args:
        question (str): The question to answer.
        image (str): Base64-encoded image string (optional).
        
    Returns:
        Dict[str, str]: The answer to the question.
    """
    print(f"DEBUG: Processing question: '{question[:100]}{'...' if len(question) > 100 else ''}'")
    print(f"DEBUG: Image provided: {image is not None}")
    
    if image:
        print("DEBUG: Getting image description...")
        image_description = get_image_description(image)
        print(f"DEBUG: Image description: '{image_description[:100]}{'...' if len(image_description) > 100 else ''}'")
        question += f" {image_description}"
        print(f"DEBUG: Updated question with image description")
    
    print("DEBUG: Getting question embedding...")
    question_embedding = get_embedding(question)
    print(f"DEBUG: Question embedding shape: {len(question_embedding)}")
    
    print("DEBUG: Loading embeddings...")
    chunks, embeddings = load_embeddings()
    
    print("DEBUG: Calculating similarities...")
    similarities = np.dot(embeddings, question_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(question_embedding))
    print(f"DEBUG: Similarities shape: {similarities.shape}")
    print(f"DEBUG: Max similarity: {np.max(similarities):.4f}, Min similarity: {np.min(similarities):.4f}")
    
    top_indices = np.argsort(similarities)[-10:][::-1]
    print(f"DEBUG: Top 10 similarity indices: {top_indices}")
    print(f"DEBUG: Top 10 similarity scores: {[similarities[i] for i in top_indices]}")
    
    top_chunks = [chunks[i] for i in top_indices]
    print(f"DEBUG: Retrieved {len(top_chunks)} top chunks")
    print(f"DEBUG: First chunk preview: '{top_chunks[0][:100]}{'...' if len(top_chunks[0]) > 100 else ''}'" if top_chunks else "DEBUG: No chunks retrieved")
    
    context = "\n".join(top_chunks)
    print(f"DEBUG: Total context length: {len(context)} characters")
    
    print("DEBUG: Generating answer...")
    response = generate_answer(question, context)
    print(f"DEBUG: Generated response length: {len(str(response))} characters")
    print(f"DEBUG: Response preview: '{str(response)[:100]}{'...' if len(str(response)) > 100 else ''}'")
    
    return response

@app.post("/api")
async def answer(request: QuestionRequest):
    """
    API endpoint to answer questions with optional image input.
    
    Args:
        request (QuestionRequest): The request containing the question and optional image.
        
    Returns:
        Dict[str, str]: The answer to the question.
    """
    print(f"DEBUG: API endpoint called with question: '{request.question[:50]}{'...' if len(request.question) > 50 else ''}'")
    print(f"DEBUG: Request has image: {request.image is not None}")
    
    try:
        response = answer_question(request.question, request.image)
        print("DEBUG: Successfully generated response")
        response = json.loads(response) 
        return response
    except Exception as e:
        print(f"DEBUG: Error occurred: {type(e).__name__}: {str(e)}")
        return {"error": f"An error occurred: {str(e)}"}

@app.get("/")
async def root():
    """
    Root endpoint to check if the server is running.
    
    Returns:
        Dict[str, str]: A simple message indicating the server is running.
    """
    print("DEBUG: Root endpoint called")
    return {"message": "Server is running. Use POST /api to ask questions."}

if __name__ == "__main__":
    import uvicorn
    print("DEBUG: Starting FastAPI server on port 8000...")
    # Run the FastAPI app
    uvicorn.run(app, port=8000)