# RAG QA Chatbot - Streamlit App with Session Management
# Save this as: app.py
# Run with: streamlit run app.py

import streamlit as st
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import tempfile
import time
from datetime import datetime
import shutil

# page configg
st.set_page_config(
    page_title="RAG QA Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# initialize session state
if "api_key" not in st.session_state:
    st.session_state.api_key = None

if "sessions" not in st.session_state:
    st.session_state.sessions = {}

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

if "session_counter" not in st.session_state:
    st.session_state.session_counter = 1

if "editing_session_id" not in st.session_state:
    st.session_state.editing_session_id = None

# helper functions


@st.cache_resource
def load_embeddings():
    """Load and cache the embedding model."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def load_llm(api_key):
    """Load the LLM with provided API key."""
    return ChatGroq(
        model_name="llama-3.1-8b-instant", temperature=0.3, groq_api_key=api_key
    )


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create chat history for a session."""
    if session_id not in st.session_state.sessions:
        return ChatMessageHistory()

    if "chat_history" not in st.session_state.sessions[session_id]:
        st.session_state.sessions[session_id]["chat_history"] = ChatMessageHistory()

    return st.session_state.sessions[session_id]["chat_history"]


def create_new_session():
    """Create a new chat session."""
    session_id = f"session_{st.session_state.session_counter}"
    st.session_state.session_counter += 1

    st.session_state.sessions[session_id] = {
        "name": f"Chat {st.session_state.session_counter - 1}",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "messages": [],
        "chat_history": ChatMessageHistory(),
        "vectorstore": None,
        "processed_files": [],
        "persist_directory": f"./chroma_db/{session_id}",
    }

    st.session_state.current_session_id = session_id
    return session_id


def delete_session(session_id):
    """Delete a session and its associated data."""
    if session_id in st.session_state.sessions:
        # Delete persisted ChromaDB
        persist_dir = st.session_state.sessions[session_id].get("persist_directory")
        if persist_dir and os.path.exists(persist_dir):
            try:
                shutil.rmtree(persist_dir)
            except Exception as e:
                st.error(f"Error deleting database: {str(e)}")

        # Delete from sessions
        del st.session_state.sessions[session_id]

        # Switch to another session or create new one
        if st.session_state.current_session_id == session_id:
            if len(st.session_state.sessions) > 0:
                st.session_state.current_session_id = list(
                    st.session_state.sessions.keys()
                )[0]
            else:
                create_new_session()


def process_pdf_files(uploaded_files, session_id):
    """Process uploaded PDF files and create/update vector store."""
    session = st.session_state.sessions[session_id]
    persist_directory = session["persist_directory"]

    # Create persist directory if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)

    all_documents = []
    total_pages = 0
    total_chunks = 0

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    embeddings = load_embeddings()

    for idx, uploaded_file in enumerate(uploaded_files):
        # Check if file already processed
        if uploaded_file.name in session["processed_files"]:
            progress_bar.progress((idx + 1) / len(uploaded_files))
            continue

        status_text.text(f"📄 Loading {uploaded_file.name}...")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        try:
            # Load PDF
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            total_pages += len(documents)

            status_text.text(f"✂️ Splitting {uploaded_file.name} into chunks...")

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            total_chunks += len(splits)
            all_documents.extend(splits)

            session["processed_files"].append(uploaded_file.name)

        finally:
            # Clean up temporary file
            os.unlink(tmp_path)

        progress_bar.progress((idx + 1) / len(uploaded_files))

    # Create or update vector store
    if all_documents:
        status_text.text("🔄 Creating vector database...")
        time.sleep(0.5)  # Brief pause for visual feedback

        if session["vectorstore"] is None:
            # Create new vectorstore with persistence
            vectorstore = Chroma.from_documents(
                documents=all_documents,
                embedding=embeddings,
                persist_directory=persist_directory,
            )
            session["vectorstore"] = vectorstore
        else:
            # Add to existing vectorsstore
            session["vectorstore"].add_documents(all_documents)

        status_text.text("✅ Processing complete!")
        time.sleep(1)

    progress_bar.empty()
    status_text.empty()

    return total_pages, total_chunks


def create_conversational_chain(vectorstore, api_key):
    """Create the complete conversational RAG chain."""
    llm = load_llm(api_key)
    retriever = vectorstore.as_retriever()

    # Context question prompt
    context_q_sys_prompt = """Given a chat history and the latest user question 
        which might reference context in the chat history, formulate a standalone 
        question which can be understood without the chat history. Do NOT answer the 
        question, just reformulate if required and otherwise return as is."""

    context_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", context_q_sys_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, context_prompt
    )

    # QA prompt
    sys_prompt = """You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, say that you don't know. Use three sentences
        at maximum and keep the answer concise.\n\n{context}"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", sys_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create chains
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Wrap with message history
    conv_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conv_rag_chain


def remove_pdf_from_session(session_id, pdf_name):
    """Remove a specific PDF from a session."""
    session = st.session_state.sessions[session_id]

    if pdf_name in session["processed_files"]:
        session["processed_files"].remove(pdf_name)

        # If no PDFs left, clear vectorstore
        if len(session["processed_files"]) == 0:
            session["vectorstore"] = None
            persist_dir = session["persist_directory"]
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir)
                os.makedirs(persist_dir, exist_ok=True)
        else:
            # Rebuild vectorstore without this PDF
            # Note: This is a simplified approach. For production, you'd want
            # to track document IDs and remove specific ones from Chroma
            st.warning(
                "PDF removed from list. To fully remove from knowledge base, delete and re-upload remaining PDFs."
            )


# setup screen/ welcome


def show_setup_screen():
    """Show the API key setup screen."""
    st.markdown(
        "<h1 style='text-align: center;'>🤖 RAG QA Chatbot</h1>", unsafe_allow_html=True
    )
    st.markdown(
        "<h3 style='text-align: center;'>Chat with Your Documents Using AI</h3>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### 🔐 Setup Your API Key")
        st.info(
            "To get started, you'll need a Groq API key. Get one free at [console.groq.com](https://console.groq.com)"
        )

        api_key_input = st.text_input(
            "Enter your Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Your API key is stored only in this session and never saved to disk",
        )

        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if st.button("🚀 Start Chatting", type="primary", use_container_width=True):
                if api_key_input and api_key_input.strip():
                    st.session_state.api_key = api_key_input.strip()
                    # Create first session
                    create_new_session()
                    st.rerun()
                else:
                    st.error("Please enter a valid API key")

        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("ℹ️ About This App"):
            st.markdown(
                """
            **Features:**
            - 📤 Upload multiple PDF documents
            - 💬 Ask questions about your documents
            - 🧠 AI-powered answers with context awareness
            - 💾 Multiple chat sessions
            - 🔒 Secure - API key stored only in memory
            
            **How it works:**
            1. Upload your PDF documents
            2. The app processes and indexes them
            3. Ask questions in natural language
            4. Get accurate answers based on your documents
            """
            )


# app

# Show setup screen if no API key
if st.session_state.api_key is None:
    show_setup_screen()
    st.stop()

# sidebar

with st.sidebar:
    st.title("🤖 RAG QA Chatbot")

    # API Key Management
    with st.expander("🔐 API Key Settings", expanded=False):
        st.text(
            "Current API Key: "
            + ("✅ Set" if st.session_state.api_key else "❌ Not Set")
        )

        if st.button("🗑️ Delete API Key", use_container_width=True):
            st.session_state.api_key = None
            st.session_state.sessions = {}
            st.session_state.current_session_id = None
            st.rerun()

    st.markdown("---")

    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        create_new_session()
        st.rerun()

    st.markdown("---")
    st.subheader("💬 Chat Sessions")

    # Display all sessions
    if len(st.session_state.sessions) == 0:
        st.info("No chat sessions yet. Create one to get started!")
    else:
        for session_id, session_data in st.session_state.sessions.items():
            is_current = session_id == st.session_state.current_session_id

            # Session container
            with st.container():
                col1, col2 = st.columns([3, 1])

                with col1:
                    # Editable session name
                    if st.session_state.editing_session_id == session_id:
                        new_name = st.text_input(
                            "Session name",
                            value=session_data["name"],
                            key=f"edit_{session_id}",
                            label_visibility="collapsed",
                        )
                        if st.button("✅", key=f"save_{session_id}"):
                            session_data["name"] = new_name
                            st.session_state.editing_session_id = None
                            st.rerun()
                    else:
                        button_type = "primary" if is_current else "secondary"
                        if st.button(
                            f"{'📌 ' if is_current else ''}{session_data['name']}",
                            key=f"session_{session_id}",
                            use_container_width=True,
                            type=button_type,
                        ):
                            st.session_state.current_session_id = session_id
                            st.rerun()

                with col2:
                    # Edit and delete buttons in a menu
                    if st.button("⋮", key=f"menu_{session_id}"):
                        st.session_state[f"show_menu_{session_id}"] = (
                            not st.session_state.get(f"show_menu_{session_id}", False)
                        )

                # Show menu options
                if st.session_state.get(f"show_menu_{session_id}", False):
                    if st.button(
                        "✏️ Rename", key=f"rename_{session_id}", use_container_width=True
                    ):
                        st.session_state.editing_session_id = session_id
                        st.rerun()

                    if st.button(
                        "🗑️ Delete", key=f"delete_{session_id}", use_container_width=True
                    ):
                        if st.session_state.get(f"confirm_delete_{session_id}", False):
                            delete_session(session_id)
                            st.session_state[f"show_menu_{session_id}"] = False
                            st.rerun()
                        else:
                            st.session_state[f"confirm_delete_{session_id}"] = True
                            st.warning(
                                "Click again to confirm deletion. This will delete all PDFs and chat history."
                            )

                # Session info
                st.caption(
                    f"📅 {session_data['created_at']} | 📄 {len(session_data['processed_files'])} PDFs"
                )
                st.markdown("<br>", unsafe_allow_html=True)

# chatinterface

# Get current session
if st.session_state.current_session_id is None:
    st.info("👈 Create a new chat session from the sidebar to get started!")
    st.stop()

current_session = st.session_state.sessions[st.session_state.current_session_id]

# Header
st.title(f"💬 {current_session['name']}")

# Check if PDFs are uploaded
if current_session["vectorstore"] is None:
    st.info("📤 Please upload PDF documents to start chatting")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.current_session_id}",
    )

    if uploaded_files:
        if st.button("🔄 Process Documents", type="primary"):
            with st.spinner("Processing your documents..."):
                pages, chunks = process_pdf_files(
                    uploaded_files, st.session_state.current_session_id
                )
                st.success(
                    f"✅ Processed {len(uploaded_files)} file(s) - {pages} pages, {chunks} chunks"
                )
                st.rerun()

    st.stop()

# Show processed files
with st.expander("📚 Uploaded Documents", expanded=False):
    for pdf_name in current_session["processed_files"]:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.text(f"📄 {pdf_name}")
        with col2:
            if st.button("❌", key=f"remove_{pdf_name}"):
                remove_pdf_from_session(st.session_state.current_session_id, pdf_name)
                st.rerun()

    st.markdown("---")

    # Allow adding more PDFs
    st.subheader("➕ Add More Documents")
    additional_files = st.file_uploader(
        "Upload additional PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"additional_uploader_{st.session_state.current_session_id}",
    )

    if additional_files:
        if st.button("🔄 Process Additional Documents"):
            with st.spinner("Processing..."):
                pages, chunks = process_pdf_files(
                    additional_files, st.session_state.current_session_id
                )
                st.success(f"✅ Added {len(additional_files)} more file(s)")
                st.rerun()

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in current_session["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    current_session["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Create conversational chain
                conv_rag_chain = create_conversational_chain(
                    current_session["vectorstore"], st.session_state.api_key
                )

                # Get response
                response = conv_rag_chain.invoke(
                    {"input": prompt},
                    config={
                        "configurable": {
                            "session_id": st.session_state.current_session_id
                        }
                    },
                )

                answer = response["answer"]
                st.markdown(answer)

                # Add assistant message
                current_session["messages"].append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                current_session["messages"].append(
                    {"role": "assistant", "content": error_msg}
                )

# Footer stats
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("💬 Messages", len(current_session["messages"]))
with col2:
    st.metric("📄 Documents", len(current_session["processed_files"]))
with col3:
    history = get_session_history(st.session_state.current_session_id)
    st.metric("🧠 Memory", len(history.messages))
