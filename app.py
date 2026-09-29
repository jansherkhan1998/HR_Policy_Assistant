import os
import re

import faiss
import fitz
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📘",
    layout="wide"
)


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title("📘 HR Policy Assistant")

st.write(
    "Ask questions about your organization's HR policies "
    "using AI-powered document search."
)


# ============================================================
# SETTINGS
# ============================================================

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"


# ============================================================
# SESSION STATE
# ============================================================

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "document_name" not in st.session_state:
    st.session_state.document_name = ""

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============================================================
# LOAD SENTENCE TRANSFORMER MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(uploaded_file):
    """
    Extract text from every page of the uploaded PDF.
    Returns a list containing page number and page text.
    """

    try:
        pdf_bytes = uploaded_file.getvalue()

        document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        pages = []

        for page_number, page in enumerate(document, start=1):

            text = page.get_text("text")

            if text and text.strip():
                pages.append(
                    {
                        "page": page_number,
                        "text": text
                    }
                )

        document.close()

        return pages, None

    except Exception as error:
        return [], str(error)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean unnecessary spaces and line breaks.
    """

    text = text.replace("\x00", " ")

    # Replace multiple whitespace characters
    # with a single space.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# TEXT CHUNKING
# ============================================================

def create_chunks(pages):
    """
    Split page text into chunks of approximately
    CHUNK_SIZE words with CHUNK_OVERLAP words overlap.

    Page number is preserved with every chunk.
    """

    chunks = []

    for page_data in pages:

        page_number = page_data["page"]

        text = clean_text(page_data["text"])

        if not text:
            continue

        words = text.split()

        start = 0

        while start < len(words):

            end = start + CHUNK_SIZE

            chunk_words = words[start:end]

            if not chunk_words:
                break

            chunk_text = " ".join(chunk_words)

            chunks.append(
                {
                    "text": chunk_text,
                    "page": page_number
                }
            )

            # Stop when we have reached the end.
            if end >= len(words):
                break

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(chunks, embedding_model):
    """
    Convert document chunks into numerical embeddings.
    """

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings.astype("float32")


# ============================================================
# CREATE FAISS INDEX
# ============================================================

def create_faiss_index(embeddings):
    """
    Create a FAISS similarity-search index.

    Inner Product is used because the embeddings
    are normalized, making it equivalent to cosine similarity.
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_chunks(
    question,
    faiss_index,
    chunks,
    embedding_model,
    top_k
):
    """
    Convert the user question into an embedding and
    retrieve the most relevant document chunks.
    """

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    question_embedding = question_embedding.astype(
        "float32"
    )

    scores, indices = faiss_index.search(
        question_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0]
    ):

        if index_position < 0:
            continue

        results.append(
            {
                "text": chunks[index_position]["text"],
                "page": chunks[index_position]["page"],
                "score": float(score)
            }
        )

    return results


# ============================================================
# GET GROQ CLIENT
# ============================================================

def get_groq_client():
    """
    Get the Groq API key from Streamlit Secrets.

    A small environment-variable fallback is included so the
    application can also work in another hosted environment.
    """

    api_key = None

    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, retrieved_chunks):
    """
    Send ONLY the retrieved policy context and user question
    to the Groq model.
    """

    client = get_groq_client()

    if client is None:
        return None, "GROQ_API_KEY is missing."

    context_parts = []

    for item in retrieved_chunks:

        context_parts.append(
            f"POLICY SOURCE - PAGE {item['page']}\n"
            f"{item['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are an HR Policy Assistant.

Your task is to answer the user's question using ONLY
the provided HR policy context.

STRICT RULES:

1. Answer only from the supplied HR policy context.
2. Do not use outside knowledge.
3. Do not invent or assume HR policies.
4. Do not fabricate page numbers or policy sections.
5. If the answer cannot be found in the supplied context,
   say exactly:

"I could not find this information in the uploaded HR policy."

6. Keep answers concise and professional.
7. When the information is available, mention the relevant
   policy page number.
8. If the policy information is ambiguous, clearly state
   that it is ambiguous instead of inventing an interpretation.
9. Do not provide legal advice.
10. Do not claim that a policy says something unless the
    retrieved context supports it.
"""

    user_prompt = f"""
HR POLICY CONTEXT:

{context}

USER QUESTION:

{question}

Answer the user's question using ONLY the HR POLICY CONTEXT.
"""

    try:

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.2,
            max_completion_tokens=1000,
            include_reasoning=False
        )

        answer = response.choices[0].message.content

        if not answer:
            return None, "The Groq model returned an empty answer."

        return answer, None

    except Exception as error:

        return None, (
            "Groq API error. Please check your API key, "
            "model availability, or deployment settings."
        )


# ============================================================
# PROCESS DOCUMENT
# ============================================================

def process_document(uploaded_file, embedding_model):
    """
    Complete document-processing pipeline:

    PDF
    ↓
    Text extraction
    ↓
    Cleaning
    ↓
    Chunking
    ↓
    Embeddings
    ↓
    FAISS
    """

    pages, extraction_error = extract_text_from_pdf(
        uploaded_file
    )

    if extraction_error:
        return None, None, None, (
            "Could not read the PDF. "
            "Please make sure it is a valid PDF file."
        )

    if not pages:
        return None, None, None, (
            "No extractable text was found in the PDF. "
            "If this is a scanned PDF, OCR may be required."
        )

    chunks = create_chunks(pages)

    if not chunks:
        return None, None, None, (
            "No usable text chunks could be created."
        )

    embeddings = create_embeddings(
        chunks,
        embedding_model
    )

    if embeddings.size == 0:
        return None, None, None, (
            "Could not generate document embeddings."
        )

    faiss_index = create_faiss_index(embeddings)

    return pages, chunks, faiss_index, None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 HR Policy")

    uploaded_file = st.file_uploader(
        "Upload HR Policy PDF",
        type=["pdf"]
    )

    st.divider()

    top_k = st.slider(
        "Retrieved policy chunks",
        min_value=3,
        max_value=8,
        value=5,
        step=1,
        help="Number of relevant policy sections retrieved for each question."
    )

    st.divider()

    if uploaded_file is not None:

        st.write("**Selected file:**")
        st.write(uploaded_file.name)

        if st.button(
            "⚙️ Process Document",
            use_container_width=True
        ):

            embedding_model = load_embedding_model()

            with st.spinner(
                "Extracting, chunking and indexing the PDF..."
            ):

                try:

                    (
                        pages,
                        chunks,
                        faiss_index,
                        error
                    ) = process_document(
                        uploaded_file,
                        embedding_model
                    )

                    if error:

                        st.error(error)

                    else:

                        st.session_state.chunks = chunks
                        st.session_state.faiss_index = faiss_index
                        st.session_state.document_name = uploaded_file.name
                        st.session_state.document_processed = True
                        st.session_state.chat_history = []

                        st.success(
                            "Document processed successfully!"
                        )

                except Exception:

                    st.error(
                        "An unexpected error occurred while "
                        "processing the document."
                    )

    if st.session_state.document_processed:

        st.divider()

        st.subheader("Document Status")

        st.success("PDF uploaded")
        st.success("Text extracted")
        st.success("Document processed")
        st.success(
            f"{len(st.session_state.chunks)} chunks created"
        )
        st.success("FAISS index ready")

        st.divider()

        if st.button(
            "🗑️ Clear / Reset Document",
            use_container_width=True
        ):

            st.session_state.chunks = []
            st.session_state.faiss_index = None
            st.session_state.document_name = ""
            st.session_state.document_processed = False
            st.session_state.chat_history = []

            st.rerun()


# ============================================================
# MAIN PAGE - DOCUMENT STATUS
# ============================================================

if uploaded_file is not None:

    st.subheader("1. Upload Policy")

    st.info(
        f"Uploaded PDF: **{uploaded_file.name}**"
    )

else:

    st.subheader("1. Upload Policy")

    st.info(
        "Please upload an HR Policy PDF using the sidebar."
    )


# ============================================================
# MAIN PAGE - STATUS
# ============================================================

st.subheader("2. Document Status")

status_col1, status_col2, status_col3, status_col4 = st.columns(4)

with status_col1:

    if uploaded_file:
        st.success("PDF uploaded")
    else:
        st.warning("Waiting for PDF")


with status_col2:

    if st.session_state.document_processed:
        st.success("Text extracted")
    else:
        st.warning("Not processed")


with status_col3:

    if st.session_state.document_processed:
        st.success(
            f"{len(st.session_state.chunks)} chunks"
        )
    else:
        st.warning("No chunks")


with status_col4:

    if st.session_state.faiss_index is not None:
        st.success("FAISS ready")
    else:
        st.warning("FAISS not ready")


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

if not st.session_state.document_processed:

    st.divider()

    st.subheader("Example Questions")

    examples = [
        "What is the annual leave policy?",
        "How many sick leaves are allowed?",
        "What is the maternity leave policy?",
        "What are the working hours?",
        "What is the resignation notice period?",
        "What is the disciplinary procedure?"
    ]

    for example in examples:
        st.write(f"• {example}")


# ============================================================
# CHAT HISTORY
# ============================================================

if st.session_state.document_processed:

    st.divider()

    st.subheader("3. Ask a Question")

    # Display previous conversation
    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            # Show sources for assistant messages.
            if (
                message["role"] == "assistant"
                and message.get("sources")
            ):

                with st.expander("📚 Sources / Retrieved Policy Sections"):

                    for source_number, source in enumerate(
                        message["sources"],
                        start=1
                    ):

                        st.markdown(
                            f"**Source {source_number} — Page {source['page']}**"
                        )

                        st.write(source["text"])

                        st.caption(
                            f"Similarity score: "
                            f"{source['score']:.3f}"
                        )


    question = st.chat_input(
        "Ask a question about the HR policy..."
    )


    # ========================================================
    # PROCESS USER QUESTION
    # ========================================================

    if question:

        question = question.strip()

        if not question:

            st.warning(
                "Please enter a question."
            )

        elif st.session_state.faiss_index is None:

            st.error(
                "The document has not been processed yet."
            )

        else:

            # Add user message
            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(question)


            embedding_model = load_embedding_model()

            with st.chat_message("assistant"):

                with st.spinner(
                    "Searching the HR policy..."
                ):

                    try:

                        retrieved_chunks = retrieve_chunks(
                            question=question,
                            faiss_index=st.session_state.faiss_index,
                            chunks=st.session_state.chunks,
                            embedding_model=embedding_model,
                            top_k=top_k
                        )

                        if not retrieved_chunks:

                            answer = (
                                "I could not find this information "
                                "in the uploaded HR policy."
                            )

                            st.markdown(answer)

                            st.session_state.chat_history.append(
                                {
                                    "role": "assistant",
                                    "content": answer,
                                    "sources": []
                                }
                            )

                        else:

                            answer, error = generate_answer(
                                question,
                                retrieved_chunks
                            )

                            if error:

                                st.error(error)

                            else:

                                st.markdown(answer)

                                st.session_state.chat_history.append(
                                    {
                                        "role": "assistant",
                                        "content": answer,
                                        "sources": retrieved_chunks
                                    }
                                )

                                with st.expander(
                                    "📚 Sources / Retrieved Policy Sections"
                                ):

                                    for source_number, source in enumerate(
                                        retrieved_chunks,
                                        start=1
                                    ):

                                        st.markdown(
                                            f"**Source {source_number} "
                                            f"— Page {source['page']}**"
                                        )

                                        st.write(
                                            source["text"]
                                        )

                                        st.caption(
                                            f"Similarity score: "
                                            f"{source['score']:.3f}"
                                        )

                    except Exception:

                        st.error(
                            "An error occurred while searching "
                            "the policy. Please try again."
                        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "HR Policy Assistant • RAG • FAISS • Sentence Transformers "
    "• PyMuPDF • Groq GPT-OSS 20B"
)
```
