import streamlit as st
import os
import time
from rag_engine import RAGEngine
from utils import (validate_pdf, sanitize_input, format_chat_history,
                   save_uploaded_file, cleanup_temp_files, format_file_size)
from dotenv import load_dotenv

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Chat AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Light Mode CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #f5f6fa !important;
    color: #1a1a2e !important;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e5e7eb !important;
    box-shadow: 2px 0 12px rgba(0,0,0,0.04);
}

section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: #f0f1ff;
    border: 2px dashed #c7d2fe;
    border-radius: 12px;
    padding: 8px;
}

.main .block-container {
    background-color: #f5f6fa !important;
    padding-top: 1rem !important;
}

.stChatMessage {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
    margin-bottom: 10px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
    animation: fadeInUp 0.35s ease-out;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

.stChatInput > div {
    background: #ffffff !important;
    border: 1.5px solid #c7d2fe !important;
    border-radius: 14px !important;
}

.stChatInput > div:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 3px 12px rgba(99,102,241,0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 18px rgba(99,102,241,0.4) !important;
}

.hero-title {
    text-align: center;
    padding: 30px 0 16px 0;
}

.hero-title h1 {
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 6px;
    color: #1a1a2e;
}

.hero-gradient {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-title p {
    color: #6b7280;
    font-size: 1rem;
}

.stat-card {
    background: #f0f1ff;
    border: 1px solid #c7d2fe;
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 8px;
}

.stat-card .stat-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #6b7280;
    margin-bottom: 3px;
}

.stat-card .stat-value {
    font-size: 1.2rem;
    font-weight: 700;
    color: #4f46e5;
}

.source-badge {
    display: inline-block;
    background: #ede9fe;
    border: 1px solid #c4b5fd;
    color: #6d28d9;
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 0.76rem;
    font-weight: 500;
    margin-top: 8px;
}

.empty-state {
    text-align: center;
    padding: 70px 30px;
}

.empty-state .icon {
    font-size: 3.5rem;
    margin-bottom: 14px;
    opacity: 0.5;
}

.empty-state h3 {
    color: #374151;
    font-weight: 600;
    margin-bottom: 8px;
    font-size: 1.2rem;
}

.empty-state p {
    color: #9ca3af;
    font-size: 0.92rem;
    max-width: 380px;
    margin: 0 auto;
    line-height: 1.6;
}

.sidebar-divider {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 16px 0;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
}

.status-ready {
    background: #dcfce7;
    border: 1px solid #bbf7d0;
    color: #16a34a;
}

.status-waiting {
    background: #fef9c3;
    border: 1px solid #fde68a;
    color: #ca8a04;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #c7d2fe; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #a5b4fc; }
</style>
""", unsafe_allow_html=True)

# ── Security: API key check ───────────────────────────────────────────────────
load_dotenv()
if not os.getenv("NVIDIA_API_KEY"):
    st.error("Missing NVIDIA_API_KEY. Please set it in your .env file.")
    st.stop()

# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "current_file_id" not in st.session_state:
    st.session_state.current_file_id = None
if "pdf_stats" not in st.session_state:
    st.session_state.pdf_stats = {}

# ── RAG Engine ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model... (first time only)")
def get_rag_engine():
    try:
        return RAGEngine()
    except Exception as e:
        st.error(f"Error initializing engine: {e}")
        st.stop()

rag_engine = get_rag_engine()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### PDF Chat AI")
    st.markdown(
        '<p style="font-size:0.82rem;color:#6b7280;margin-top:-8px;line-height:1.6;">'
        'Upload any PDF and ask questions in plain language.'
        'Your document is split into smart chunks, embedded into a searchable index,'
        'and only the most relevant sections are retrieved to answer each question —'
        'giving you fast, accurate, and context-aware responses every time.'
        '</p>',
        unsafe_allow_html=True
    )
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown("##### Upload Document")
    uploaded_file = st.file_uploader(
        "Drop PDF here",
        type=["pdf"],
        label_visibility="collapsed",
        help="Maximum 200 MB"
    )

    if uploaded_file is not None:
        if st.session_state.current_file_id != uploaded_file.file_id:
            st.session_state.pdf_processed = False

        if not st.session_state.pdf_processed:
            try:
                validate_pdf(uploaded_file)
                temp_path = save_uploaded_file(uploaded_file)

                progress_bar = st.progress(0, text="Reading PDF...")
                pages_count, chunks_count = rag_engine.load_and_embed_pdf(
                    temp_path,
                    progress_callback=lambda p, t: progress_bar.progress(
                        min(10 + int(p * 85), 95), text=t
                    )
                )
                progress_bar.progress(100, text="Done!")
                time.sleep(0.4)
                progress_bar.empty()

                st.session_state.pdf_processed = True
                st.session_state.current_file_id = uploaded_file.file_id
                st.session_state.pdf_stats = {
                    "filename": uploaded_file.name,
                    "pages": pages_count,
                    "chunks": chunks_count,
                    "size": format_file_size(uploaded_file.size)
                }
                cleanup_temp_files()

                if pages_count > 0:
                    st.success(f"Processed {pages_count} pages into {chunks_count} chunks!")
                else:
                    st.error("No text found in this PDF.")

            except Exception as e:
                st.error(f"{e}")
                cleanup_temp_files()
        else:
            st.markdown('<span class="status-pill status-ready">Document Ready</span>',
                        unsafe_allow_html=True)
    else:
        if st.session_state.pdf_processed or st.session_state.pdf_stats:
            st.session_state.pdf_processed = False
            st.session_state.current_file_id = None
            st.session_state.pdf_stats = {}
        st.markdown('<span class="status-pill status-waiting">Awaiting Upload</span>',
                    unsafe_allow_html=True)

    if st.session_state.pdf_stats:
        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
        st.markdown("##### Document Stats")
        stats = st.session_state.pdf_stats

        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">File</div>
            <div class="stat-value" style="font-size:0.88rem;color:#374151;">{stats.get("filename","")}</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Pages</div>
                <div class="stat-value">{stats.get("pages", 0)}</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Chunks</div>
                <div class="stat-value">{stats.get("chunks", 0)}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Size</div>
            <div class="stat-value" style="font-size:1rem;">{stats.get("size","")}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
        st.markdown("##### 💡 Suggested Questions")
        
        suggestions = [
            "Summarize this document in 3 paragraphs.",
            "What are the 5 key takeaways?",
            "Extract any important dates or deadlines.",
            "What is the main conclusion of this PDF?",
        ]
        
        for q in suggestions:
            if st.button(q, key=f"btn_{q}", use_container_width=True):
                # When a suggestion is clicked, it's added to the session state
                # and handled in the main chat loop
                st.session_state.messages.append({"role": "user", "content": q})
                with st.spinner("Processing suggestion..."):
                    try:
                        res = rag_engine.query(q)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": res["answer"],
                            "sources": res["sources"]
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center;color:#9ca3af;font-size:0.7rem;">Streamlit · NVIDIA NIM · FAISS</p>',
        unsafe_allow_html=True
    )

# ── Main Area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-title">
    <h1>👨‍💻 <span class="hero-gradient">Chat With Any PDF</span> 📄</h1>
    <p>Upload a document and ask questions — powered by AI retrieval</p>
</div>
""", unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            pages_str = ", ".join(str(p) for p in msg["sources"])
            st.markdown(
                f'<span class="source-badge">Source: Page {pages_str}</span>',
                unsafe_allow_html=True
            )

if not st.session_state.messages:
    if st.session_state.pdf_processed:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">💬</div>
            <h3>Your document is ready!</h3>
            <p>Start asking questions about your PDF below.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">📄</div>
            <h3>No document uploaded yet</h3>
            <p>Upload a PDF in the sidebar to get started. Ask about any reports, papers, or manuals.</p>
        </div>""", unsafe_allow_html=True)

if prompt := st.chat_input("Ask anything about your PDF..."):
    clean_prompt = sanitize_input(prompt)

    if not clean_prompt:
        st.error("Invalid input. Please enter a valid question.")
    elif not st.session_state.pdf_processed:
        st.error("Please upload and process a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": clean_prompt})
        with st.chat_message("user"):
            st.markdown(clean_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching document and generating answer..."):
                try:
                    res = rag_engine.query(clean_prompt)
                    answer = res["answer"]
                    sources = res["sources"]

                    st.markdown(answer)
                    if sources:
                        pages_str = ", ".join(str(p) for p in sources)
                        st.markdown(
                            f'<span class="source-badge">Source: Page {pages_str}</span>',
                            unsafe_allow_html=True
                        )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Error querying AI: {e}")
