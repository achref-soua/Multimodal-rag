# Multimodal RAG System

A **multimodal retrieval-augmented generation (RAG)** system capable of processing both text and images from PDF documents. This system allows users to ask questions about PDF content and receive context-aware answers with both textual and visual references.

---

## Features

- Extract **text and images** from PDFs
- Unified **embedding system using CLIP**
- **Multimodal retrieval** with FAISS
- Question answering powered by **Groq's Llama model**
- Clean **class-based architecture** for easy maintenance
- Streamlit interface for **interactive usage**
- Robust **error handling and file management**

---

## Setup

1. **Clone the repository**  
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

2. **Install dependencies**  
```bash
pip install -r requirements.txt
```

3. **Create a `.env` file** with your API keys:  
```env
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
```

4. **Run the app**  
```bash
streamlit run app.py
```

---

## Usage

1. Upload a **PDF document**  
2. Wait for **processing** to complete  
3. Ask **questions** about the document content  
4. View **answers** along with retrieved **context** (text and images)

---

## Shoutout

A big shoutout to [**krishnaik06**](https://github.com/krishnaik06) for his amazing tutorials and inspiration in the field of AI and RAG systems! 🚀

---

## Contributing

Feel free to **fork the repo** and submit **pull requests**. Issues and suggestions are always welcome!

---

## License

This project is licensed under the MIT License.