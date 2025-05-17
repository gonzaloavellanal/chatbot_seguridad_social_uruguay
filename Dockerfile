# Imagen base optimizada para LLMs y compatibilidad FAISS
FROM python:3.10-slim

# Evita prompts y mejora logs
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Crea y entra al directorio de trabajo
WORKDIR /app

# Instala dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Crea el directorio para PDFs
RUN mkdir pdfs

# Copia archivos del proyecto al contenedor
COPY backend.py ./
COPY streamlit_app.py ./
COPY pdfs/*.pdf ./pdfs/
COPY requirements.txt ./
COPY .env ./

# Instala las dependencias de Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expone el puerto para Streamlit
EXPOSE 8501

# Variables de entorno para Streamlit
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Comando para ejecutar la aplicación Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
