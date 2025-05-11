# streamlit_app.py

import streamlit as st
from backend import responder_pregunta

st.set_page_config(page_title="Chatbot Ley de Seguridad Social", page_icon="⚖️")
st.title("💬 Chatbot sobre Ley de Seguridad Social (Uruguay)")
st.write("Preguntá sobre la Ley 20.130 (reforma de la seguridad social).")

if "historial" not in st.session_state:
    st.session_state.historial = []

pregunta = st.chat_input("Escribí tu pregunta aquí...")

if pregunta:
    st.chat_message("user").markdown(pregunta)
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            respuesta = responder_pregunta(pregunta)
            st.markdown(respuesta)
    st.session_state.historial.append((pregunta, respuesta))