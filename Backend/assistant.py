from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import numpy as np
import faiss
import os
from PyPDF2 import PdfReader

api_key = "uzm2ETbml7L9ZxTMaiUgcQVsFjv4Dzen"
client = MistralClient(api_key=api_key)

# Ruta al archivo PDF local
pdf_path = os.path.join('Backend/Documento_para_ChatBot.pdf')

# Función para leer el archivo PDF
def read_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ''.join([page.extract_text() for page in reader.pages])
        return text
    except Exception as e:
        print(f"Error: No se pudo leer el archivo PDF. {e}")
        return None

# Función para obtener embeddings de texto
def get_text_embeddings(input):
    embeddings_batch_response = client.embeddings(
        model="mistral-embed",
        input=input
    )
    return [embedding.data[0].embedding for embedding in embeddings_batch_response]

# Función para guardar texto en un archivo de texto plano
def save_text_to_file(text, file_path):
    with open(file_path, 'w') as f:
        f.write(text)

# Leer el archivo PDF
text = read_pdf(pdf_path)
if text:
    # Guardar el contenido en un archivo de texto plano (opcional)
    text_path = os.path.join('Backend/Documento_para_ChatBot.txt')
    save_text_to_file(text, text_path)

    # Dividir el texto en chunks
    chunk_size = 2048
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Obtener embeddings de los chunks de texto en lotes
    text_embeddings = np.array(get_text_embeddings(chunks))

    # Crear índice FAISS y agregar embeddings
    d = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(text_embeddings)

    # Pregunta de ejemplo
    question = "Dame información acerca del acuerdo N°3"
    question_embedding = np.array(get_text_embeddings([question]))

    # Buscar en el índice FAISS
    D, I = index.search(question_embedding, k=2)

    # Recuperar chunks relevantes
    retrieved_chunks = [chunks[i] for i in I[0]]

    # Crear prompt para Mistral
    prompt = f"""
    La información del contexto se encuentra a continuación.
    ---------------------
    {' '.join(retrieved_chunks)}
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
