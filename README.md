# ⚖️ Chatbot sobre la Ley de Seguridad Social (Uruguay) – Ley 20.130

Este es un chatbot legal desarrollado con un modelo de lenguaje LLaMA (a través de Hugging Face) y un sistema RAG (Retrieval-Augmented Generation).  
El modelo responde preguntas exclusivamente en base al texto completo de la **Ley 20.130 de Reforma de la Seguridad Social en Uruguay (2023)**.

---

## 🤖 ¿Qué hace?

- Permite hacer preguntas en lenguaje natural sobre la ley.
- Recupera los artículos relevantes de forma semántica (utilizando vectorización con FAISS).
- Genera respuestas claras y completas usando el modelo LLaMA.
- Siempre indica si la información está o no en la ley, y recomienda consultar a un experto si la pregunta excede el alcance del modelo.

---

## 🗃 Estructura del proyecto

| Archivo                 | Función principal                                        |
|-------------------------|----------------------------------------------------------|
| `backend.py`            | Lógica del modelo (carga de PDF, FAISS, LLaMA, RAG)      |
| `streamlit_app.py`      | Interfaz de usuario con Streamlit (chatbot conversacional) |
| `Ley_20130_2023.pdf`    | Fuente de datos legal (ley completa)                     |
| `requirements.txt`      | Lista de dependencias del proyecto                       |
| `.env`                  | Contiene el token privado de Hugging Face (no subir)     |
| `.gitignore`            | Ignora archivos temporales y sensibles                   |

---

## ⚙️ Requisitos

- Python 3.9 o superior.
- Un token de Hugging Face con acceso al modelo LLaMA.
- Los paquetes listados en `requirements.txt`.

---

## 🔐 Configuración

Antes de ejecutar la app, asegúrate de tener un archivo `.env` en la raíz del proyecto con el siguiente contenido:

HF_TOKEN=tu_token_aquí


---

## 🚀 Cómo ejecutar la app

1. Clona el repositorio o ábrelo en Codespaces:
   ```bash
   git clone https://github.com/gonzaloavellanal/chatbot_seguridad_social_uruguay
   cd chatbot-seguridad-social-uruguay
   ```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Ejecuta la app:
```bash
streamlit run streamlit_app.py
```
