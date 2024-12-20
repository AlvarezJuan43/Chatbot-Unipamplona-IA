from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain import PromptTemplate, LLMChain
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import numpy as np
import faiss
import os
from Backend.helpers import read_pdf, get_text_embedding, initialize_faiss_index

api_key = "uzm2ETbml7L9ZxTMaiUgcQVsFjv4Dzen"
client = MistralClient(api_key=api_key)

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

def preprocess_text(text):
    return ' '.join(text.split())

# Crear plantilla de prompt para LangChain
template = """
La información relevante del documento se encuentra a continuación:
---------------------
{context}
---------------------
Por favor, responda con la información relevante del documento en el formato más apropiado. Si la respuesta incluye elementos listados o enumerados, preséntelos en forma de lista.
Consulta: {question}
Respuesta:
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

@app.post("/chat")
async def chat(message: Message):
    user_message = message.message

    question_embeddings = np.array([get_text_embedding(user_message)])
    D, I = index.search(question_embeddings, k=3)
    retrieved_chunks = [chunks[i] for i in I.tolist()[0]]
    
    # Preprocesar los chunks recuperados
    processed_chunks = [preprocess_text(chunk) for chunk in retrieved_chunks]

    # Crear una cadena LLM con LangChain utilizando Mistral
    def run_mistral(prompt):
        messages = [ChatMessage(role="user", content=prompt)]
        chat_response = client.chat(model="mistral-medium-latest", messages=messages)
        return chat_response.choices[0].message.content

    # Ejecutar la cadena
    context = ' '.join(processed_chunks)
    response = run_mistral(prompt.format(context=context, question=user_message))

    return {"response": response}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
