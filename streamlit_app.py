# streamlit_app.py

import sys
import os

print(f"Python path: {sys.path}")
print(f"Directorio actual: {os.getcwd()}")
print(f"Archivos en directorio: {os.listdir('.')}")

try:
    print("Intentando importar backend...")
    from backend import responder_pregunta
    print("Importación exitosa.")
except Exception as e:
    import traceback
    print(f"Error al importar: {str(e)}")
    print(traceback.format_exc())

# Resto del código de Streamlit

import streamlit as st
from backend import responder_pregunta

st.set_page_config(page_title="Chatbot Ley de Seguridad Social", page_icon="⚖️")
st.title("💬 Chatbot sobre Ley de Seguridad Social (Uruguay)")
st.write("Preguntá sobre la Ley 20.130 (reforma de la seguridad social).")

if "historial" not in st.session_state:
    st.session_state.historial = []

for mensaje_usuario, mensaje_bot in st.session_state.historial:
    st.chat_message("user").markdown(mensaje_usuario)
    st.chat_message("assistant").markdown(mensaje_bot)

pregunta = st.chat_input("Escribí tu pregunta aquí...")

if pregunta:
    st.chat_message("user").markdown(pregunta)
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                respuesta = responder_pregunta(pregunta)
            except Exception as e:
                respuesta = "⚠️ Ocurrió un error al procesar la respuesta."
                print(f"Error: {e}")
            st.markdown(respuesta)
    st.session_state.historial.append((pregunta, respuesta))


