import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class RAGEngine:
    def __init__(self):
        self.api_key = os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "NVIDIA_API_KEY is not set in environment variables. "
                "Please create a .env file with your key."
            )

        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key
        )
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        self.index_path = "./faiss_index/"

    # ─── PDF Loading & Embedding ──────────────────────────────────────────

    def load_and_embed_pdf(self, pdf_path: str, progress_callback=None) -> tuple:
        """
        Extract text page-by-page, split into chunks, embed in batches,
        and store in a FAISS vector store on disk.

        Args:
            pdf_path: Path to the PDF file.
            progress_callback: Optional callable(progress_float, status_text).
                               progress_float is 0.0–1.0.

        Returns:
            (page_count, chunk_count) tuple.
        """
        # ── Step 1: Load pages ────────────────────────────────────────────
        if progress_callback:
            progress_callback(0.0, "📖 Loading PDF pages...")

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        if not pages:
            return 0, 0

        # Tag each page with a 1-indexed page number
        for i, page in enumerate(pages):
            page.metadata["page"] = i + 1

        if progress_callback:
            progress_callback(0.15, f"📄 Loaded {len(pages)} pages. Splitting text...")

        # ── Step 2: Split into chunks ─────────────────────────────────────
        # Use larger chunks for big documents for better context
        chunk_size = 1000 if len(pages) > 50 else 500
        chunk_overlap = 100 if len(pages) > 50 else 50

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(pages)

        if not chunks:
            return len(pages), 0

        if progress_callback:
            progress_callback(0.30, f"✂️ Created {len(chunks)} chunks. Embedding...")

        # ── Step 3: Batch embed into FAISS ────────────────────────────────
        # For large PDFs (many chunks), we process in batches to avoid
        # memory spikes and to report accurate progress.
        batch_size = 500
        total_chunks = len(chunks)

        if total_chunks <= batch_size:
            # Small document — embed in one shot
            if progress_callback:
                progress_callback(0.50, "🧠 Generating embeddings...")
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            # Large document — batch embed with progress
            first_batch = chunks[:batch_size]
            if progress_callback:
                progress_callback(0.35, f"🧠 Embedding batch 1/{(total_chunks // batch_size) + 1}...")
            self.vector_store = FAISS.from_documents(first_batch, self.embeddings)

            for start in range(batch_size, total_chunks, batch_size):
                end = min(start + batch_size, total_chunks)
                batch_num = (start // batch_size) + 1
                total_batches = (total_chunks // batch_size) + 1
                progress = 0.35 + (0.55 * (start / total_chunks))

                if progress_callback:
                    progress_callback(progress, f"🧠 Embedding batch {batch_num}/{total_batches}...")

                batch = chunks[start:end]
                batch_store = FAISS.from_documents(batch, self.embeddings)
                self.vector_store.merge_from(batch_store)

        if progress_callback:
            progress_callback(0.92, "💾 Saving index to disk...")

        # ── Step 4: Save index ────────────────────────────────────────────
        os.makedirs(self.index_path, exist_ok=True)
        self.vector_store.save_local(self.index_path)

        if progress_callback:
            progress_callback(1.0, "✅ Done!")

        return len(pages), len(chunks)

    # ─── Query ────────────────────────────────────────────────────────────

    def query(self, question: str, k: int = 4) -> dict:
        """
        Retrieve relevant chunks, build a prompt, and query the NVIDIA LLM.

        Returns:
            {"answer": str, "sources": [int]}
        """
        if not self.vector_store:
            if os.path.exists(self.index_path) and os.path.exists(
                os.path.join(self.index_path, "index.faiss")
            ):
                self.vector_store = FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                return {
                    "answer": "No document has been processed yet. Please upload a PDF first.",
                    "sources": []
                }

        # Retrieve top-k relevant chunks
        docs = self.vector_store.similarity_search(question, k=k)
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])

        # Deduplicate & sort source page numbers
        sources = sorted(set(
            doc.metadata.get("page", 0) for doc in docs
        ))

        system_prompt = (
            "You are a helpful assistant. Answer ONLY from the provided context. "
            "If the answer is not in the context, say 'I cannot find this in the document.' "
            "Do not make up information."
        )

        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            f"Answer the question based only on the context above."
        )

        response = self.client.chat.completions.create(
            model="meta/llama-4-maverick-17b-128e-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=1024
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": sources
        }
