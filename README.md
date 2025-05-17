# ⚖️ Chatbot sobre la Ley de Seguridad Social (Uruguay)

Este chatbot legal utiliza tecnología de vanguardia para responder consultas sobre la legislación de seguridad social uruguaya. Desarrollado con LLaMA y RAG (Retrieval-Augmented Generation), proporciona respuestas precisas basadas en la documentación legal oficial.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url-here)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📚 Contenido
- [Características](#-características)
- [Tecnologías](#-tecnologías)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Configuración](#-configuración)
- [Contribución](#-contribución)
- [Licencia](#-licencia)

## ✨ Características

- 🤖 Interfaz conversacional intuitiva
- 📝 Respuestas basadas en documentación legal oficial
- 🔍 Búsqueda semántica avanzada
- 📊 Citación precisa de artículos relevantes
- ⚡ Procesamiento eficiente de consultas
- 🔒 Manejo seguro de información sensible

## 🛠 Tecnologías

- **Frontend**: Streamlit
- **Backend**: Python, LangChain
- **Modelos**: LLaMA (Hugging Face)
- **Vectorización**: FAISS
- **Procesamiento de PDF**: pdfminer.six
- **Gestión de Dependencias**: pip

## 🗃 Estructura del Proyecto

```text
chatbot_seguridad_social_uruguay/
├── backend.py           # Lógica del modelo y procesamiento
├── streamlit_app.py     # Interfaz de usuario
├── pdfs/               # Directorio de documentos legales
│   └── Ley_20130_2023.pdf
├── requirements.txt    # Dependencias del proyecto
├── .env               # Variables de entorno (no compartir)
└── README.md          # Documentación
```

## ⚙️ Requisitos

- Python 3.9 o superior
- Token de Hugging Face con acceso a LLaMA
- 4GB RAM mínimo recomendado
- Conexión a Internet estable

## 📥 Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/gonzaloavellanal/chatbot_seguridad_social_uruguay
cd chatbot_seguridad_social_uruguay
```

2. **Crear y activar entorno virtual (opcional pero recomendado)**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
.\venv\Scripts\activate  # Windows
```

3. **Instalar dependencias**
```bash
python -m pip install -r requirements.txt
```

## 🚀 Uso

1. **Configurar variables de entorno**
```bash
echo "HF_TOKEN=tu_token_aquí" > .env
```

2. **Ejecutar la aplicación**
```bash
streamlit run streamlit_app.py
```

## ⚙️ Configuración Avanzada

### Variables de Entorno
```env
HF_TOKEN=tu_token_de_huggingface
```

### Configuración de Modelos
El chatbot utiliza:
- Embedding: `intfloat/multilingual-e5-small`
- LLM: Modelo LLaMA optimizado

## 🤝 Contribución

Las contribuciones son bienvenidas. Por favor:

1. Haz Fork del proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 👥 Autores

- **Gonzalo Avellanal - Juan Ignacio Briozzo - Gabriel Lopez**

## 🙏 Agradecimientos

- A la comunidad de Hugging Face por proporcionar acceso a modelos de lenguaje avanzados
- A los desarrolladores de LangChain por sus herramientas de procesamiento de lenguaje natural
- A la comunidad de Streamlit por facilitar la creación de interfaces web para aplicaciones de ML

---
