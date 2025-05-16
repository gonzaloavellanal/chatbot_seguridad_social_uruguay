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
 
# Copia archivos del proyecto al contenedor
COPY backend.py ./main.py
COPY Ley_20130_2023.pdf ./
COPY .env ./
COPY requirements.txt ./
 
# Instala las dependencias de Python
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
 
# Expone el puerto 7860 (por si querés exponer interfaz más adelante)
EXPOSE 7860
 
# Comando para ejecutar el backend
CMD ["python", "main.py"]
