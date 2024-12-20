from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

from Backend.helpers import read_pdf, get_text_embedding, initialize_faiss_index, run_langchain_with_mistral
import os

app = FastAPI()

# Permitir solicitudes CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ruta al archivo PDF local
pdf_path = os.path.join('Backend/Documento_para_ChatBot.pdf')

class Message(BaseModel):
    message: str

# Inicialización de la base de datos vectorial
index, chunks = initialize_faiss_index(pdf_path)

@app.post("/chat")
async def chat(message: Message):
    user_message = message.message

    question_embeddings = np.array([get_text_embedding(user_message)])
    D, I = index.search(question_embeddings, k=3)
    retrieved_chunks = [chunks[i] for i in I.tolist()[0]]

    # Ejecutar la cadena LLM con LangChain y Mistral
    context = ' '.join(retrieved_chunks)
    response = run_langchain_with_mistral(context, user_message)

    return {"response": response}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
