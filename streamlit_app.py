# streamlit_app.py

import streamlit as st
import os
from pathlib import Path
from backend import initialize_chatbot, ChatbotBackend
from dotenv import load_dotenv
import nest_asyncio
import asyncio

# Arreglar el event loop
try:
    nest_asyncio.apply()
except Exception as e:
    st.error(f"Error configurando nest_asyncio: {str(e)}")

# Asegurarse de que hay un event loop
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Configuración de la página
st.set_page_config(
    page_title="Chatbot Seguridad Social Uruguay",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función para cargar el token de HuggingFace
def load_huggingface_token():
    load_dotenv()
    token = os.getenv('HF_TOKEN')
    if not token:
        st.error("No se encontró el token de HuggingFace. Por favor, configura la variable HF_TOKEN en el archivo .env")
        st.stop()
    return token

# Función para inicializar el chatbot
def init_chatbot():
    if 'chatbot' not in st.session_state:
        try:
            # Cargar token
            hf_token = load_huggingface_token()
            
            # Buscar PDFs en el directorio pdfs
            pdf_dir = Path("pdfs")
            if not pdf_dir.exists():
                st.error("No se encontró el directorio 'pdfs'. Por favor, crea el directorio y agrega los archivos PDF.")
                st.stop()
            
            pdf_files = list(pdf_dir.glob("*.pdf"))
            if not pdf_files:
                st.error("No se encontraron archivos PDF en el directorio 'pdfs'.")
                st.stop()
            
            # Mostrar información de carga
            with st.spinner("Inicializando el chatbot... Este proceso puede tomar unos minutos."):
                st.session_state.chatbot = initialize_chatbot(
                    pdf_files=[str(pdf) for pdf in pdf_files],
                    hf_token=hf_token
                )
                st.session_state.messages = []
                
            st.success("¡Chatbot inicializado correctamente!")
            
        except Exception as e:
            st.error(f"Error inicializando el chatbot: {str(e)}")
            st.stop()

# Función para mostrar el historial de chat
def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Función para procesar la pregunta del usuario
def process_user_input(prompt):
    if not prompt:
        return
    
    # Agregar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar y mostrar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Procesando tu pregunta..."):
            try:
                response = st.session_state.chatbot.answer_question(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error generando respuesta: {str(e)}")

# Interfaz principal
def main():
    st.title("💬 Chatbot de Seguridad Social Uruguay")
    
    # Sidebar con información
    with st.sidebar:
        st.title("Información")
        st.markdown("""
        ### Sobre este chatbot
        Este asistente está especializado en responder preguntas sobre:
        - Ley 20.130 de Seguridad Social
        - Otros documentos relacionados
        
        ### Consejos para mejores resultados:
        1. Sea específico en sus preguntas
        2. Pregunte sobre un tema a la vez
        3. Use términos técnicos cuando sea posible
        
        ### Archivos disponibles:
        """)
        
        # Mostrar archivos PDF disponibles
        pdf_dir = Path("pdfs")
        if pdf_dir.exists():
            pdfs = list(pdf_dir.glob("*.pdf"))
            for pdf in pdfs:
                st.markdown(f"- {pdf.name}")
        
        # Botón para reiniciar el chat
        if st.button("Reiniciar Chat"):
            st.session_state.messages = []
            st.experimental_rerun()

    # Inicializar chatbot
    init_chatbot()

    # Mostrar historial del chat
    display_chat_history()

    # Input del usuario
    if prompt := st.chat_input("Hazme una pregunta sobre seguridad social..."):
        process_user_input(prompt)

if __name__ == "__main__":
    main()