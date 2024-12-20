from mistralai.client import MistralClient
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from PyPDF2 import PdfReader
import numpy as np
import faiss
import os

api_key = "uzm2ETbml7L9ZxTMaiUgcQVsFjv4Dzen"
client = MistralClient(api_key=api_key)

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

# Inicialización de la base de datos vectorial
def initialize_faiss_index(pdf_path):
    text = read_pdf(pdf_path)
    chunks = []
    index = None
    if text:
        chunk_size = 2048
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])
        d = text_embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(text_embeddings)
    return index, chunks

# Función para crear y ejecutar una cadena LLM con LangChain utilizando Mistral
def run_langchain_with_mistral(context, question):
    # Crear plantilla de prompt para LangChain
    template = """
    La información del contexto se encuentra a continuación.
    ---------------------
    {context}
    ---------------------
    Responda la información del contexto y no el conocimiento previo, sea lo más breve posible sin extenderse demasiado.
    Consulta: {question}
    Respuesta:
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)

    # Crear una cadena LLM con LangChain
    llm = OpenAI(api_key=api_key)
    chain = LLMChain(llm=llm, prompt=prompt)

    # Ejecutar la cadena
    response = chain.run({"context": context, "question": question})
    return response
