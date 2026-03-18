import os
import json
import asyncio
import aiohttp
import logging
import pickle
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
from datetime import datetime
from typing import List, Dict

# -----------------------
# CONFIG
# -----------------------
DATA_SOURCES = {
    "linkedin": "https://www.linkedin.com/in/freemangoja/",
    "medium": "https://medium.com/@freemangoja",
    "ailysis": "https://ailysis.io",
    "speaker": "https://world.aiacceleratorinstitute.com/location/agenticaitoronto/speaker/freemangoja"
}

CHUNK_SIZE = 100  # tokens/words per chunk
CHUNK_OVERLAP = 20  # overlap tokens
KB_DIR = "kb"

MODEL_NAME = "all-MiniLM-L6-v2"  # semantic retrieval optimized
embedder = SentenceTransformer(MODEL_NAME)

os.makedirs(KB_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO)

# -----------------------
# UTILITY FUNCTIONS
# -----------------------
async def fetch(session: aiohttp.ClientSession, url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        async with session.get(url, headers=headers, timeout=10) as response:
            html = await response.text()
            soup = BeautifulSoup(html, "html.parser")
            for s in soup(["script", "style"]):
                s.decompose()
            text = " ".join(soup.stripped_strings)
            return text
    except Exception as e:
        logging.error(f"Failed to scrape {url}: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def create_document_chunks(text: str, source: str, doc_type: str = "general", metadata: Dict = None) -> List[Dict]:
    metadata = metadata or {}
    chunks = chunk_text(text)
    return [
        {
            "text": chunk,
            "source": source,
            "type": doc_type,
            "metadata": metadata
        }
        for chunk in chunks
    ]

# -----------------------
# MAIN BUILD FUNCTION
# -----------------------
async def build_knowledge_base():
    async with aiohttp.ClientSession() as session:
        all_docs = []

        for source_name, url in DATA_SOURCES.items():
            logging.info(f"Scraping {source_name}...")
            text = await fetch(session, url)
            if not text:
                continue

            # Determine document type heuristically
            doc_type = "project" if "github" in url or "portfolio" in url else "article"

            # Add metadata example
            metadata = {
                "title": source_name.capitalize(),
                "date_fetched": str(datetime.now())
            }

            docs = create_document_chunks(text, source_name, doc_type, metadata)
            all_docs.extend(docs)

        if not all_docs:
            logging.warning("No documents collected. Exiting.")
            return

        logging.info("Generating embeddings...")
        texts = [d["text"] for d in all_docs]
        embeddings = embedder.encode(texts, batch_size=16, show_progress_bar=True)

        dim = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dim)
        id_map = faiss.IndexIDMap(index)
        ids = list(range(len(embeddings)))
        id_map.add_with_ids(np.array(embeddings).astype("float32"), np.array(ids))

        # Save vector index and docs
        faiss.write_index(id_map, os.path.join(KB_DIR, "index.faiss"))
        with open(os.path.join(KB_DIR, "docs.pkl"), "wb") as f:
            pickle.dump(all_docs, f)

        logging.info(f"Knowledge base built with {len(all_docs)} chunks.")

# -----------------------
# ENTRY POINT
# -----------------------
if __name__ == "__main__":
    import numpy as np
    asyncio.run(build_knowledge_base())