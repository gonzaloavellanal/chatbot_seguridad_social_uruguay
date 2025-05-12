import os
import re
import textwrap
from dotenv import load_dotenv
from pdfminer.high_level import extract_text
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Cargar variables de entorno
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Token HF_TOKEN no encontrado en .env")
login(HF_TOKEN)

# Cargar y limpiar PDF
def limpiar_encabezados_pies(texto):
    patrones = [
        r'http\S+', r'Página\s*\d+\s*de\s*\d+', r'\?Ver.*', r'Ver Imagen.*',
        r'Ver en esta norma.*', r'\(\*\)Notas:.*', r'Reglamentada por:.*',
        r'^\s*$', r'Ver Base Jurisprudencia Nacional.*', r'\?',
        r'Referencias al artículo.*',
    ]
    for patron in patrones:
        texto = re.sub(patron, '', texto, flags=re.MULTILINE)
    return re.sub(r'\n\s*\n', '\n\n', texto)

def extraer_estructura_completa(path_to_pdf):
    texto_limpio = limpiar_encabezados_pies(extract_text(path_to_pdf))
    patron_titulo = re.compile(r'^\s*(T[ÍI]TULO\s+[IVXLCDM]+\s*[-–]\s+[^.,)\n]+)$', re.MULTILINE)
    patron_seccion = re.compile(r'^\s*(SECCI[ÓO]N\s+[IVXLCDM]+\s*[-–]?\s+.+?)\s*$', re.MULTILINE)
    patron_capitulo = re.compile(r'^\s*(CAP[ÍI]TULO\s+(?:[IVXLCDM]+|ÚNICO)\s*[-–]?\s+.+?)\s*$', re.MULTILINE)
    patron_articulo = re.compile(r'^\s*Art[íi]culo\s+\d+\s*$', re.MULTILINE)

    posiciones = []
    texto = texto_limpio
    for regex, tipo in [(patron_titulo, 'titulo'), (patron_seccion, 'seccion'),
                        (patron_capitulo, 'capitulo'), (patron_articulo, 'articulo')]:
        for m in regex.finditer(texto): posiciones.append((m.start(), tipo, m.group().strip()))
    posiciones.sort(key=lambda x: x[0])

    resultado, metadatos = [], {"titulo": "", "seccion": "", "capitulo": ""}
    for i in range(len(posiciones)):
        inicio, tipo, texto_tipo = posiciones[i]
        fin = posiciones[i+1][0] if i+1 < len(posiciones) else len(texto)
        contenido = texto[inicio:fin].strip()
        if tipo in metadatos:
            metadatos[tipo] = texto_tipo
        elif tipo == 'articulo':
            resultado.append({
                "titulo": metadatos["titulo"], "seccion": metadatos["seccion"],
                "capitulo": metadatos["capitulo"], "articulo": texto_tipo, "contenido": contenido
            })
    return resultado

# Preparar artículos
pdf_path = "Ley_20130_2023.pdf"
articulos = extraer_estructura_completa(pdf_path)
documents = [Document(page_content=a["contenido"], metadata=a) for a in articulos]

# Vector store
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
faiss_index = FAISS.from_documents(documents, embedding_model)
retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# LLM + RAG
pipe = pipeline("text-generation",
                model="meta-llama/Llama-3.2-3B-Instruct",
                torch_dtype="auto",
                device_map="auto",
                temperature=0.1,
                do_sample=True,
                repetition_penalty=1.1,
                return_full_text=False,
                max_new_tokens=400)

llm = HuggingFacePipeline(pipeline=pipe)

prompt_template = """
<|start_header_id|>user<|end_header_id|>
Eres un asistente experto en la ley 20.130 de seguridad social en Uruguay.
Responde de forma clara y completa usando el siguiente contexto legal.
Incluí los artículos usados si aplica. Si no encontrás respuesta, decí "No lo sé".
Siempre terminá recomendando consultar a un experto.
Pregunta: {question}
Contexto: {context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

rag_chain = (
    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
     "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser())


def responder_pregunta(pregunta):
    return rag_chain.invoke(pregunta)