import fitz  # PyMuPDF
from langchain_core.documents import Document
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from langchain_groq import ChatGroq
from langchain.schema.messages import HumanMessage
import os
import base64
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class MultimodalRAG:
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", use_safetensors=True
        )
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.clip_model.eval()

        # Initialize LLM
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",
            temperature=1,
        )

        self.vector_store = None
        self.image_data_store = {}
        self.all_docs = []

    def embed_image(self, image_data):
        """Embed image using CLIP"""
        if isinstance(image_data, str):  # If path
            image = Image.open(image_data).convert("RGB")
        else:  # If PIL Image
            image = image_data

        inputs = self.clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.squeeze().numpy()

    def embed_text(self, text):
        """Embed text using CLIP."""
        inputs = self.clip_processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,  # CLIP's max token length
        )
        with torch.no_grad():
            features = self.clip_model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.squeeze().numpy()

    def process_pdf(self, pdf_path):
        """Process PDF and create vector store"""
        doc = fitz.open(pdf_path)
        all_docs = []
        all_embeddings = []
        image_data_store = {}
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

        for i, page in enumerate(doc):
            # Process text
            text = page.get_text()
            if text.strip():
                temp_doc = Document(
                    page_content=text, metadata={"page": i, "type": "text"}
                )
                text_chunks = splitter.split_documents([temp_doc])

                for chunk in text_chunks:
                    embedding = self.embed_text(chunk.page_content)
                    all_embeddings.append(embedding)
                    all_docs.append(chunk)

            # Process images
            for img_index, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    image_id = f"page_{i}_img_{img_index}"

                    buffered = io.BytesIO()
                    pil_image.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    image_data_store[image_id] = img_base64

                    embedding = self.embed_image(pil_image)
                    all_embeddings.append(embedding)

                    image_doc = Document(
                        page_content=f"[Image: {image_id}]",
                        metadata={"page": i, "type": "image", "image_id": image_id},
                    )
                    all_docs.append(image_doc)

                except Exception as e:
                    print(f"Error processing image {img_index} on page {i}: {e}")
                    continue

        doc.close()

        # Create vector store
        embeddings_array = np.array(all_embeddings)
        vector_store = FAISS.from_embeddings(
            text_embeddings=[
                (doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)
            ],
            embedding=None,
            metadatas=[doc.metadata for doc in all_docs],
        )

        self.vector_store = vector_store
        self.image_data_store = image_data_store
        self.all_docs = all_docs

    def retrieve_multimodal(self, query, k=5):
        """Unified retrieval using CLIP embeddings"""
        query_embedding = self.embed_text(query)
        results = self.vector_store.similarity_search_by_vector(
            embedding=query_embedding, k=k
        )
        return results

    def create_multimodal_message(self, query, retrieved_docs):
        """Create a message with both text and images"""
        content = []
        content.append({"type": "text", "text": f"Question: {query}\\n\\nContext:\\n"})

        text_docs = [
            doc for doc in retrieved_docs if doc.metadata.get("type") == "text"
        ]
        image_docs = [
            doc for doc in retrieved_docs if doc.metadata.get("type") == "image"
        ]

        if text_docs:
            text_context = "\\n\\n".join(
                [
                    f"[Page {doc.metadata['page']}]: {doc.page_content}"
                    for doc in text_docs
                ]
            )
            content.append(
                {"type": "text", "text": f"Text excerpts:\\n{text_context}\\n"}
            )

        for doc in image_docs:
            image_id = doc.metadata.get("image_id")
            if image_id and image_id in self.image_data_store:
                content.append(
                    {
                        "type": "text",
                        "text": f"\\n[Image from page {doc.metadata['page']}]:\\n",
                    }
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{self.image_data_store[image_id]}"
                        },
                    }
                )

        content.append(
            {
                "type": "text",
                "text": "\\n\\nPlease answer the question based on the provided text and images.",
            }
        )
        return HumanMessage(content=content)

    def query(self, question):
        """Main query function"""
        context_docs = self.retrieve_multimodal(question, k=5)
        message = self.create_multimodal_message(question, context_docs)
        response = self.llm.invoke([message])
        return response.content, context_docs
