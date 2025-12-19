# --- Windows fix for 'pwd' import issue in LangChain ---
import sys, types

if sys.platform == "win32" and "pwd" not in sys.modules:
    sys.modules["pwd"] = types.SimpleNamespace(getpwuid=lambda x: None)
# --------------------------------------------------------

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings.base import Embeddings
import google.generativeai as genai


# --- Simple, Reliable TF-IDF Embeddings ---
class TFIDFEmbeddings(Embeddings):
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=384)
        self.fitted = False

    def embed_documents(self, texts):
        # texts is already a list of strings from FAISS
        if not texts:
            return [[0.0] * 384]

        if not self.fitted:
            self.vectorizer.fit(texts)
            self.fitted = True

        vectors = self.vectorizer.transform(texts).toarray()
        # Pad to consistent size
        result = []
        for v in vectors:
            if len(v) < 384:
                result.append(list(v) + [0] * (384 - len(v)))
            else:
                result.append(list(v[:384]))
        return result

    def embed_query(self, text):
        if not self.fitted:
            return [0.0] * 384

        vector = self.vectorizer.transform([text]).toarray()[0]
        if len(vector) < 384:
            return list(vector) + [0] * (384 - len(vector))
        else:
            return list(vector[:384])


# --- Load environment variables ---
load_dotenv()

# --- Configure Gemini API ---
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("⚠️ GEMINI_API_KEY not found in .env file!")
    st.info("Please create a .env file with: GEMINI_API_KEY=your_api_key_here")
    st.stop()

genai.configure(api_key=api_key)

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="⚖️ LawConnect", page_icon="⚖️", layout="wide")
st.title("⚖️ LawConnect - Legal Document Q&A Bot")
st.markdown("Upload your legal document and ask questions directly!")
st.caption("🌏 Supports English, Tamil, and other languages")

# --- Initialize session state ---
if 'processed_file' not in st.session_state:
    st.session_state.processed_file = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'messages' not in st.session_state:
    st.session_state.messages = []


# --- Helper function to detect greetings ---
def is_greeting(text):
    greetings = ['hi', 'hello', 'hey', 'hi there', 'hello there', 'good morning',
                 'good afternoon', 'good evening', 'greetings', 'sup', 'yo',
                 'whats up', "what's up", 'howdy', 'வணக்கம்', 'नमस्ते']
    text_lower = text.lower().strip()
    return text_lower in greetings or any(text_lower.startswith(g) for g in greetings)


# --- Sidebar for file upload ---
with st.sidebar:
    st.header("📁 Document Upload")
    uploaded_file = st.file_uploader("Upload your file", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        # Check if this is a new file
        if st.session_state.processed_file != uploaded_file.name:
            st.session_state.processed_file = uploaded_file.name
            st.session_state.messages = []
            st.session_state.chat_history = []

            with st.spinner("Processing document..."):
                try:
                    # Save uploaded file
                    os.makedirs("uploads", exist_ok=True)
                    file_path = os.path.join("uploads", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Load document
                    st.info("📄 Loading document...")
                    if uploaded_file.name.endswith(".pdf"):
                        loader = PyPDFLoader(file_path)
                    elif uploaded_file.name.endswith(".docx"):
                        loader = Docx2txtLoader(file_path)
                    else:
                        loader = TextLoader(file_path, encoding='utf-8')

                    documents = loader.load()

                    if not documents:
                        st.error("No content found in document!")
                        st.session_state.processed_file = None
                        st.stop()

                    # Split into chunks
                    st.info("✂️ Splitting document into chunks...")
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                    )
                    text_chunks = text_splitter.split_documents(documents)

                    if not text_chunks:
                        st.error("Could not split document into chunks!")
                        st.session_state.processed_file = None
                        st.stop()

                    # Use TF-IDF Embeddings
                    st.info("🔍 Creating embeddings...")
                    embeddings = TFIDFEmbeddings()

                    # Create FAISS vector store
                    st.info("🗄️ Creating vector database...")
                    vectorstore_dir = "vectorstore"
                    os.makedirs(vectorstore_dir, exist_ok=True)

                    vectorstore = FAISS.from_documents(text_chunks, embeddings)
                    vectorstore.save_local(vectorstore_dir)

                    # Initialize memory and LLM
                    st.info("🤖 Initializing AI model...")
                    memory = ConversationBufferMemory(
                        memory_key="chat_history",
                        return_messages=True,
                        output_key='answer'
                    )

                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash",
                        temperature=0.3,
                        google_api_key=api_key
                    )

                    # Create retrieval chain
                    retriever = vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 3}
                    )

                    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=retriever,
                        memory=memory,
                        return_source_documents=True,
                        verbose=False
                    )

                    st.success(f"✅ '{uploaded_file.name}' processed successfully!")
                    st.info(f"📊 Document split into {len(text_chunks)} chunks")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.processed_file = None
                    import traceback

                    with st.expander("🔍 View Error Details"):
                        st.code(traceback.format_exc())
        else:
            st.success(f"✅ Using '{uploaded_file.name}'")

    # Language support info
    with st.expander("🌍 Supported Languages"):
        st.markdown("""
        **Full Support:**
        - 🇬🇧 English
        - 🇮🇳 Tamil (தமிழ்)
        - 🇮🇳 Hindi (हिंदी)
        - And 100+ other languages via Gemini

        **Note:** Upload documents in any language 
        and ask questions in English or the document's language!
        """)

    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        if st.session_state.qa_chain:
            st.session_state.qa_chain.memory.clear()
        st.rerun()

# --- Main chat interface ---
if st.session_state.processed_file is None:
    st.info("⬆️ Please upload a PDF, DOCX, or TXT file from the sidebar to get started.")

    # Quick start instructions
    with st.expander("📖 How to Use LawConnect"):
        st.markdown("""
        ### Getting Started:
        1. **Upload a document** (PDF, DOCX, or TXT)
        2. **Wait for processing** (instant!)
        3. **Ask questions** in any language

        ### Example Questions:

        **English:** "What is this document about?"

        **Tamil:** "இந்த ஆவணம் எதைப் பற்றியது?"

        **Hindi:** "यह दस्तावेज़ किस बारे में है?"

        ### Features:
        - Multilingual document support
        - Ask in English about Tamil/Hindi docs
        - Instant processing
        - Source references
        """)
else:
    st.subheader("💬 Chat with Your Document")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question in any language..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            # Check if it's a greeting
            if is_greeting(prompt):
                answer = """Hello! 👋 I'm **LawConnect**, your multilingual legal document assistant.

📌 **What I can do:**
- Understand documents in English, Tamil, Hindi, and 100+ languages
- Answer your questions in any language
- Show you exact source references

**Try asking:**
- "What is this document about?"
- "Who are the parties involved?"
- "Summarize the key points"

**Ready to help! What would you like to know?**"""

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            else:
                # Normal RAG processing
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.qa_chain({"question": prompt})
                        answer = response["answer"]

                        st.markdown(answer)

                        # Show source references
                        if response.get("source_documents"):
                            with st.expander("📄 View Source References"):
                                for idx, doc in enumerate(response["source_documents"], 1):
                                    st.markdown(f"**Source {idx}:**")
                                    st.text(doc.page_content[:400] + "...")
                                    if hasattr(doc, 'metadata') and doc.metadata:
                                        st.caption(f"Metadata: {doc.metadata}")
                                    st.divider()

                        st.session_state.messages.append({"role": "assistant", "content": answer})

                    except Exception as e:
                        error_message = f"❌ Error: {str(e)}"
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
                        import traceback

                        with st.expander("🔍 View Error Details"):
                            st.code(traceback.format_exc())

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
        ⚖️ z AI 
        Supports 100+ languages including Tamil, Hindi, English
    </div>
    """,
    unsafe_allow_html=True
)
