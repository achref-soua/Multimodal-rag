import streamlit as st
import tempfile
import os
from multimodal_rag import MultimodalRAG

# Set page config
st.set_page_config(page_title="Multimodal RAG", page_icon="📄", layout="wide")

# Initialize session state
if "rag" not in st.session_state:
    st.session_state.rag = None
if "processed" not in st.session_state:
    st.session_state.processed = False

st.title("Multimodal RAG System")
st.write("Upload a PDF document to ask questions about its content (text and images)")

# File upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name

    # Process PDF
    if not st.session_state.processed:
        with st.spinner("Processing PDF..."):
            rag = MultimodalRAG()
            rag.process_pdf(pdf_path)
            st.session_state.rag = rag
            st.session_state.processed = True
        st.success("PDF processed successfully!")

    # Question input
    question = st.text_input("Enter your question:")

    if question:
        with st.spinner("Searching for answer..."):
            answer, context_docs = st.session_state.rag.query(question)

            # Display answer
            st.subheader("Answer:")
            st.write(answer)

            # Display context
            st.subheader("Retrieved Context:")
            for doc in context_docs:
                with st.expander(
                    f"Page {doc.metadata.get('page', 'N/A')} - {doc.metadata.get('type', 'unknown')}"
                ):
                    if doc.metadata.get("type") == "text":
                        st.text(doc.page_content)
                    else:
                        image_id = doc.metadata.get("image_id")
                        if (
                            image_id
                            and image_id in st.session_state.rag.image_data_store
                        ):
                            st.image(
                                f"data:image/png;base64,{st.session_state.rag.image_data_store[image_id]}"
                            )

    # Clean up
    os.unlink(pdf_path)
else:
    st.info("Please upload a PDF file to get started.")
