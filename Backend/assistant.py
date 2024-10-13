from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import requests
import numpy as np
import faiss
import os
from PyPDF2 import PdfReader

api_key = "yFrIll8Xpub6EPz8dchwsFPeIhL8bKyJ"
client = MistralClient(api_key=api_key)

# Ruta al archivo PDF local
pdf_path = os.path.join('Backend/Documento_para_ChatBot.pdf')

# Función para leer el archivo PDF
def read_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
        return text
    except Exception as e:
        print(f"Error: No se pudo leer el archivo PDF. {e}")
        return None

# Función para obtener embeddings de texto
def get_text_embedding(input):
    embeddings_batch_response = client.embeddings(
        model="mistral-embed",
        input=input
    )
    return embeddings_batch_response.data[0].embedding

# Función para guardar texto en un archivo de texto plano
def save_text_to_file(text, file_path):
    with open(file_path, 'w') as f:
        f.write(text)

# Leer el archivo PDF
text = read_pdf(pdf_path)
if text:
    # Imprimir el contenido para verificar
    #print(text)

    # Guardar el contenido en un archivo de texto plano (opcional)
    text_path = os.path.join('Backend/Documento_para_ChatBot.txt')
    save_text_to_file(text, text_path)

    # Dividir el texto en chunks
    chunk_size = 2048
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Obtener embeddings de los chunks de texto
    text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])

    # Crear índice FAISS y agregar embeddings
    d = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(text_embeddings)

    # Pregunta de ejemplo
    question = "Dame información acerca del acuerdo N°3"
    question_embeddings = np.array([get_text_embedding(question)])
    
    # Buscar en el índice FAISS
    D, I = index.search(question_embeddings, k=2)
    
    # Recuperar chunks relevantes
    retrieved_chunk = [chunks[i] for i in I.tolist()[0]]

    # Crear prompt para Mistral
    prompt = f"""
    La información del contexto se encuentra a continuación.
    ---------------------
    {retrieved_chunk}
    ---------------------
    Responda la información del contexto y no el conocimiento previo, sea lo más breve posible sin extenderse demasiado.
    Consulta: {question}
    Respuesta:
    """

    # Función para ejecutar Mistral
    def run_mistral(user_message, model="mistral-medium-latest"):
        messages = [
            ChatMessage(role="user", content=user_message)
        ]
        chat_response = client.chat(
            model=model,
            messages=messages
        )
        return chat_response.choices[0].message.content

    # Ejecutar Mistral con el prompt
    response = run_mistral(prompt)
    print(response)
