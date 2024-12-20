from mistralai.client import MistralClient
import numpy as np
import faiss
import os
from PyPDF2 import PdfReader
from Backend.helpers import read_pdf, get_text_embedding, initialize_faiss_index, run_langchain_with_mistral

api_key = "uzm2ETbml7L9ZxTMaiUgcQVsFjv4Dzen"
client = MistralClient(api_key=api_key)

# Ruta al archivo PDF local
pdf_path = os.path.join('Backend/Documento_para_ChatBot.pdf')

# Leer el archivo PDF
text = read_pdf(pdf_path)
if text:
    # Guardar el contenido en un archivo de texto plano (opcional)
    text_path = os.path.join('Backend/Documento_para_ChatBot.txt')
    with open(text_path, 'w') as f:
        f.write(text)

    # Dividir el texto en chunks
    chunk_size = 2048
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Obtener embeddings de los chunks de texto en lotes
    text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])

    # Crear índice FAISS y agregar embeddings
    d = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(text_embeddings)

    # Pregunta de ejemplo
    question = "Dame información acerca del acuerdo N°3"
    question_embedding = np.array(get_text_embedding([question]))

    # Buscar en el índice FAISS
    D, I = index.search(question_embedding, k=2)

    # Recuperar chunks relevantes
    retrieved_chunks = [chunks[i] for i in I[0]]

    # Ejecutar la cadena LLM con LangChain y Mistral
    context = ' '.join(retrieved_chunks)
    response = run_langchain_with_mistral(context, question)
    print(response)
