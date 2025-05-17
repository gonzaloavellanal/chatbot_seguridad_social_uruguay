import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any
from contextlib import contextmanager
import torch
import gc
from dotenv import load_dotenv
from pdfminer.high_level import extract_text
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.documents import Document
from huggingface_hub import login
from transformers import pipeline, AutoTokenizer
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotBackend:
    def __init__(self, pdf_files: List[str]):
        """
        Initialize chatbot with multiple PDF files
        pdf_files: List of PDF file paths
        """
        # Inicialización de variables de instancia
        self.pdf_files = [Path(pdf) for pdf in pdf_files]
        self.chunk_size = 500  # Para procesamiento en lotes
        self.processed_files = set()  # Para tracking de archivos procesados
        self.vector_store = None
        self.embedding_model = None
        self.tokenizer = None
        self.pipe = None
        self.retriever = None
        self.prompt = None
        self.rag_chain = None

        # Inicialización de componentes
        logger.info("Iniciando carga del chatbot...")
        self.load_environment()
        self.initialize_components()
        logger.info("Chatbot inicializado correctamente")

    def load_environment(self) -> None:
        """Load environment variables and authenticate"""
        self.hf_token = os.environ.get('HF_TOKEN')
        if not self.hf_token:
            raise ValueError("HF_TOKEN not found in environment")
        login(self.hf_token)

    @contextmanager
    def gpu_memory_management(self):
        """Context manager for GPU memory management"""
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def optimize_memory(self):
        """Optimize memory usage"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Monitoreo de memoria (opcional)
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
            logger.info(f"GPU Memory allocated: {memory_allocated:.2f} MB")

    def clean_text(self, text: str) -> str:
        """Clean text from headers and footers"""
        patterns = [
            r'http\S+', r'Página\s*\d+\s*de\s*\d+', r'\?Ver.*', r'Ver Imagen.*',
            r'Ver en esta norma.*', r'\(\*\)Notas:.*', r'Reglamentada por:.*',
            r'^\s*$', r'Ver Base Jurisprudencia Nacional.*', r'\?',
            r'Referencias al artículo.*',
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)
        return re.sub(r'\n\s*\n', '\n\n', text)

    def extraer_estructura_completa(self, texto):
        """Extract complete structure from text"""
        patron_titulo = re.compile(r'^\s*(T[ÍI]TULO\s+[IVXLCDM]+\s*[-–]\s+[^.,)\n]+)$', re.MULTILINE)
        patron_seccion = re.compile(r'^\s*(SECCI[ÓO]N\s+[IVXLCDM]+\s*[-–]?\s+.+?)\s*$', re.MULTILINE)
        patron_capitulo = re.compile(r'^\s*(CAP[ÍI]TULO\s+(?:[IVXLCDM]+|ÚNICO)\s*[-–]?\s+.+?)\s*$', re.MULTILINE)
        patron_articulo = re.compile(r'^\s*Art[íi]culo\s+\d+\s*$', re.MULTILINE)

        posiciones = []
        for regex, tipo in [(patron_titulo, 'titulo'), (patron_seccion, 'seccion'),
                           (patron_capitulo, 'capitulo'), (patron_articulo, 'articulo')]:
            for m in regex.finditer(texto):
                posiciones.append((m.start(), tipo, m.group().strip()))
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
                    "titulo": metadatos["titulo"],
                    "seccion": metadatos["seccion"],
                    "capitulo": metadatos["capitulo"],
                    "articulo": texto_tipo,
                    "contenido": contenido
                })
        return resultado

    def extract_document_structure_single_file(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract structure from a single PDF file"""
        try:
            logger.info(f"Opening file: {pdf_path}")
            with open(pdf_path, 'rb') as file:
                text = extract_text(file)

            if not text.strip():
                logger.warning(f"No text extracted from {pdf_path}")
                return []

            logger.info(f"Cleaning text from {pdf_path}")
            cleaned_text = self.clean_text(text)

            logger.info(f"Extracting structure from {pdf_path}")
            articles = self.extraer_estructura_completa(cleaned_text)

            if not articles:
                logger.warning(f"No articles found in {pdf_path}")
                return []

            # Add source document to metadata
            for article in articles:
                article['source_document'] = pdf_path.name

            logger.info(f"Successfully extracted {len(articles)} articles from {pdf_path}")
            return articles

        except FileNotFoundError:
            logger.error(f"File not found: {pdf_path}")
            return []
        except PermissionError:
            logger.error(f"Permission denied accessing file: {pdf_path}")
            return []
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return []

    def extract_document_structure(self) -> List[Dict[str, str]]:
        """Extract structured content from multiple PDF files"""
        all_articles = []

        for pdf_path in self.pdf_files:
            if not pdf_path.exists():
                logger.warning(f"PDF file not found: {pdf_path}")
                continue

            if pdf_path.name in self.processed_files:
                logger.info(f"Skipping already processed file: {pdf_path.name}")
                continue

            try:
                logger.info(f"Processing {pdf_path.name}...")
                articles = self.extract_document_structure_single_file(pdf_path)

                if articles:  # Solo agregar si se obtuvieron artículos
                    all_articles.extend(articles)
                    self.processed_files.add(pdf_path.name)
                    logger.info(f"Successfully processed {pdf_path.name} - {len(articles)} articles found")
                else:
                    logger.warning(f"No articles found in {pdf_path.name}")

            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {str(e)}")
                continue

        if not all_articles:
            raise ValueError("No articles were successfully extracted from any PDF")

        logger.info(f"Total articles extracted: {len(all_articles)}")
        logger.info(f"Processed files: {', '.join(self.processed_files)}")

        return all_articles

    def process_documents(self) -> List[Document]:
        """Process documents with proper error handling"""
        try:
            articles = self.extract_document_structure()
            return [Document(page_content=a["contenido"], metadata=a) for a in articles]
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise

    def process_documents_in_batches(self) -> List[Document]:
        """Process documents in batches to manage memory"""
        all_documents = []
        batch_size = self.chunk_size

        for pdf_path in self.pdf_files:
            try:
                articles = self.extract_document_structure_single_file(pdf_path)

                # Process in batches
                for i in range(0, len(articles), batch_size):
                    batch = articles[i:i + batch_size]
                    docs = [Document(page_content=a["contenido"],
                                  metadata={**a, "source": pdf_path.name})
                          for a in batch]
                    all_documents.extend(docs)
                    self.optimize_memory()

            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {str(e)}")
                continue

        return all_documents

    def initialize_components(self) -> None:
        """Initialize ML components with resource management"""
        try:
            with self.gpu_memory_management():
                logger.info("Initializing embedding model...")
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name="intfloat/multilingual-e5-small",
                    model_kwargs={'device': 'cpu'}  # Forzar CPU
                )

                logger.info("Initializing language model...")
                # Cambiar a un modelo más ligero
                model_name = "microsoft/phi-2"  # Modelo más pequeño
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)

                self.pipe = pipeline(
                    "text-generation",
                    model=model_name,
                    torch_dtype=torch.float32,  # Usar float32 en lugar de float16
                    device_map="auto",
                    temperature=0.3,
                    do_sample=True,
                    repetition_penalty=1.2,
                    return_full_text=False,
                    max_new_tokens=600,
                    model_kwargs={'low_cpu_mem_usage': True}  # Optimización de memoria
                )

                logger.info("Setting up RAG chain...")
                self.setup_rag_chain()
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise

    def setup_rag_chain(self) -> None:
        """Setup RAG chain with proper error handling"""
        try:
            documents = self.process_documents()
            self.vector_store = FAISS.from_documents(documents, self.embedding_model)
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}  # Aumentamos de 3 a 5 para obtener más contexto
            )

            # Prompt mejorado con instrucciones más específicas
            prompt_template = """
            <|start_header_id|>user<|end_header_id|>
            Eres un asistente experto en leyes de seguridad social en Uruguay.
            Tienes acceso a la siguiente legislación:
            {available_laws}

            Instrucciones específicas:
            1. Lee cuidadosamente el contexto proporcionado
            2. Identifica información numérica específica (porcentajes, montos, plazos)
            3. Cuando se pregunte por distribuciones o aportes, SIEMPRE incluye los porcentajes exactos
            4. Cita textualmente las partes relevantes de los artículos
            5. Estructura la información de manera clara y concisa

            Pregunta: {question}
            Contexto: {context}

            Responde siguiendo esta estructura:
            1. Respuesta directa con números y porcentajes específicos
            2. Cita textual de los artículos relevantes
            3. Explicación adicional si es necesaria
            4. Ley(es) consultada(s)
            5. Recomendación de consultar a un experto para casos específicos

            Si no encuentras información específica sobre porcentajes o montos, indica claramente "No encuentro la información específica sobre los porcentajes/montos en el contexto proporcionado."
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """

            self.prompt = PromptTemplate(
                input_variables=["available_laws", "context", "question"],
                template=prompt_template
            )

            # Modificar el RAG chain para incluir las leyes disponibles
            available_laws = "\n".join([f"- {pdf.name}" for pdf in self.pdf_files])

            self.rag_chain = (
                {"context": self.retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
                 "question": RunnablePassthrough(),
                 "available_laws": lambda _: available_laws}
                | self.prompt
                | HuggingFacePipeline(pipeline=self.pipe)
                | StrOutputParser()
            )
        except Exception as e:
            logger.error(f"Error setting up RAG chain: {str(e)}")
            raise

    def answer_question(self, question: str) -> str:
        """Answer questions with error handling"""
        try:
            with self.gpu_memory_management():
                logger.info(f"Processing question: {question}")
                response = self.rag_chain.invoke(question)
                return response
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return "Lo siento, hubo un error procesando tu pregunta. Por favor, intenta nuevamente."

    def save_state(self, filepath: str = "chatbot_state.pkl"):
        """Save the chatbot's vector store state"""
        try:
            self.vector_store.save_local(filepath)
            logger.info(f"Chatbot state saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving chatbot state: {str(e)}")

    @classmethod
    def load_from_state(cls, pdf_files: List[str], filepath: str = "chatbot_state.pkl"):
        """Load a chatbot instance from a saved state"""
        try:
            instance = cls(pdf_files)
            instance.vector_store = FAISS.load_local(filepath, instance.embedding_model)
            logger.info(f"Chatbot state loaded from {filepath}")
            return instance
        except Exception as e:
            logger.error(f"Error loading chatbot state: {str(e)}")
            raise

# Función de ayuda para inicializar el chatbot
def initialize_chatbot(pdf_files: List[str], hf_token: str = None):
    """
    Inicializa el chatbot con los archivos PDF proporcionados
    """
    if hf_token:
        os.environ['HF_TOKEN'] = hf_token
    
    try:
        chatbot = ChatbotBackend(pdf_files)
        return chatbot
    except Exception as e:
        logger.error(f"Error initializing chatbot: {str(e)}")
        raise

if __name__ == "__main__":
    # Código de prueba
    load_dotenv()
    pdf_files = ["Ley_20130_2023.pdf"]  # Ejemplo
    chatbot = initialize_chatbot(pdf_files)
    
    # Prueba simple
    response = chatbot.answer_question("¿Cuál es la distribución de aportes entre BPS y AFAP?")
    print(response)