import asyncio
import os
import requests
import json
import uuid
import time
from typing import List, Dict, Any, Optional

import numpy as np
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from bs4 import BeautifulSoup
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

import redis
from redis.commands.json.path import Path
from redis.commands.search.query import Query as RSQuery
from redis.commands.search.field import TextField, TagField, VectorField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

from sentence_transformers import SentenceTransformer

# ========= ENV & CLIENTS =========
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(REDIS_URL)

# Embedding model
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
embedder = SentenceTransformer(EMBED_MODEL_NAME)
EMBED_DIM = embedder.get_sentence_embedding_dimension()

# Gemini API URL
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
)

# Redis index
INDEX_NAME = "idx:pages"
KEY_PREFIX = "doc:"  # semua key JSON akan diawali ini

app = FastAPI(title="Crawl + Gemini + Redis VectorDB (RAG)")

# ========= UTILS =========
def clean_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    for s in soup(["script", "style"]):
        s.extract()
    text = soup.get_text(separator=" ")
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return " ".join(chunk for chunk in chunks if chunk)[:5000]


def gemini_request(prompt: str) -> str:
    payload = {"contents": [{"parts": [{"text": prompt}]}]}  # simple text prompt
    try:
        resp = requests.post(
            GEMINI_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        return f"Error: {resp.status_code}, {resp.text}"
    except Exception as e:
        return f"Error: {str(e)}"


def embed_text(text: str) -> np.ndarray:
    # returns float32 vector
    v = embedder.encode(text, normalize_embeddings=True)
    if not isinstance(v, np.ndarray):
        v = np.array(v)
    return v.astype(np.float32)


def to_bytes(vec: np.ndarray) -> bytes:
    return vec.tobytes(order="C")


def ensure_redis_index():
    """Buat index ON JSON, sekali saja."""
    try:
        # Coba info; kalau gagal berarti belum ada
        redis_client.ft(INDEX_NAME).info()
        return
    except Exception:
        pass

    schema = (
        TextField("$.id", as_name="id"),
        TextField("$.site", as_name="site"),
        TextField("$.url", as_name="url"),
        TextField("$.kind", as_name="kind"),  # "page" atau "final"
        TextField("$.summary", as_name="summary"),
        NumericField("$.created_at", as_name="created_at"),
        # VectorField untuk JSON path "$.vector"
        VectorField(
            "$.vector",
            "HNSW",
            {
                "TYPE": "FLOAT32",
                "DIM": EMBED_DIM,
                "DISTANCE_METRIC": "COSINE",
                "M": 16,
                "EF_CONSTRUCTION": 200,
            },
            as_name="vector",
        ),
    )

    definition = IndexDefinition(
        prefix=[KEY_PREFIX], index_type=IndexType.JSON
    )

    redis_client.ft(INDEX_NAME).create_index(schema, definition=definition)


def save_doc_to_redis(
    *,
    site: str,
    url: Optional[str],
    kind: str,  # "page" | "final"
    summary: str,
) -> str:
    """Simpan dokumen JSON + vector ke Redis dengan UUID; return doc_id (uuid)."""
    doc_id = str(uuid.uuid4())
    key = f"{KEY_PREFIX}{doc_id}"

    vec = embed_text(summary)
    payload = {
        "id": doc_id,
        "site": site,              # domain atau root url
        "url": url or "",
        "kind": kind,              # "page" atau "final"
        "summary": summary,
        "created_at": int(time.time()),
        # vector diset terpisah sebagai binary
    }

    # Simpan JSON dulu
    redis_client.json().set(key, Path.root_path(), payload)
    # Set vector binary di path $.vector
    redis_client.execute_command(
        "JSON.SET", key, "$.vector", '"{}"'.format("")  # placeholder
    )
    redis_client.execute_command(
        "JSON.NUMINCRBY", key, "$.created_at", 0  # memastikan field numeric ada
    )
    # Set langsung sebagai raw blob (gunakan low-level because JSON supports strings;
    # kita simpan ke Redis key hash tambahan untuk vector agar efisien):
    # Alternatif yang lebih rapi: simpan vector di JSON sebagai blob base64.
    # DI BAWAH INI: simpan vector sebagai HASH field terpisah agar pasti binary.
    redis_client.hset(f"{key}:vec", mapping={"vector": to_bytes(vec)})

    # Sinkronkan hash vector ke JSON field vector agar bisa di-index:
    # RediSearch v2.8+ bisa index VECTOR dari HASH atau JSON. Agar JSON path bekerja,
    # kita perlu set JSON field sebagai BINARY. Trik: gunakan JSON.SET dengan "\uXXXX" akan rusak.
    # Solusi stabil: indeks dari HASH, lalu gunakan INDEX yang berbeda.
    # --- Namun untuk kesederhanaan, kita lakukan pendekatan: index VECTOR dari HASH ---
    # Jadi kita buat index kedua khusus HASH (opsional). Untuk memudahkan, kita ganti:
    # -> Gunakan HASH sebagai storage utama agar VECTOR bisa langsung diindex sebagai BLOB.
    # Untuk menjaga konsistensi, di bawah ada versi HASH store.

    return doc_id


# Versi HASH yang konsisten untuk VECTOR indexing (direkomendasikan)
HASH_INDEX = "idx:pages_hash"
HASH_PREFIX = "hdoc:"

def ensure_hash_index():
    try:
        redis_client.ft(HASH_INDEX).info()
        return
    except Exception:
        pass

    schema = (
        TextField("id"),
        TextField("site"),
        TextField("url"),
        TextField("kind"),
        TextField("summary"),
        NumericField("created_at"),
        VectorField(
            "vector",
            "HNSW",
            {
                "TYPE": "FLOAT32",
                "DIM": EMBED_DIM,
                "DISTANCE_METRIC": "COSINE",
                "M": 16,
                "EF_CONSTRUCTION": 200,
            },
        ),
    )

    definition = IndexDefinition(prefix=[HASH_PREFIX], index_type=IndexType.HASH)
    redis_client.ft(HASH_INDEX).create_index(schema, definition=definition)


def save_doc_hash(
    *,
    site: str,
    url: Optional[str],
    kind: str,
    summary: str,
) -> str:
    doc_id = str(uuid.uuid4())
    key = f"{HASH_PREFIX}{doc_id}"
    vec = embed_text(summary)

    redis_client.hset(
        key,
        mapping={
            "id": doc_id,
            "site": site,
            "url": url or "",
            "kind": kind,
            "summary": summary,
            "created_at": int(time.time()),
            "vector": to_bytes(vec),
        },
    )
    return doc_id


def semantic_search(
    query: str,
    top_k: int = 5,
    site: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """KNN search pada HASH index."""
    qvec = embed_text(query)
    base = f"*=>[KNN {top_k} @vector $vec AS score]"
    if site:
        base = f"@site:({site}) {base}"

    q = RSQuery(base).return_fields(
        "id", "site", "url", "kind", "summary", "created_at", "score"
    ).sort_by("score").paging(0, top_k).dialect(2)

    res = redis_client.ft(HASH_INDEX).search(q, query_params={"vec": to_bytes(qvec)})

    hits = []
    for doc in res.docs:
        hits.append(
            {
                "id": doc.id.replace(HASH_PREFIX, ""),
                "site": doc.site,
                "url": doc.url,
                "kind": doc.kind,
                "summary": doc.summary,
                "created_at": int(doc.created_at) if doc.created_at else None,
                "score": float(doc.score),
            }
        )
    return hits


# ========= CRAWL PIPELINE =========
async def crawl_and_analyze(url: str, depth: int, pages: int):
    ensure_hash_index()

    # Tentukan "site" dari root url (sederhana: pakai url input)
    site = url

    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=depth,
            include_external=False,
            max_pages=pages,
        ),
        verbose=True,
    )

    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(url=url, config=config)

        page_entries = []
        for i, page in enumerate(results, 1):
            if hasattr(page, "html") and page.html:
                clean_content = clean_html(page.html)
                summary = gemini_request(
                    f"Summarize this web page from {page.url}:\n\n{clean_content}"
                )

                doc_id = save_doc_hash(site=site, url=page.url, kind="page", summary=summary)

                page_entries.append(
                    {"uuid": doc_id, "url": page.url, "summary": summary}
                )

        # Final summary dari seluruh page
        combined_text = "\n\n".join([p["summary"] for p in page_entries])
        final_summary = gemini_request(
            "Create a concise overall summary of these page summaries:\n\n" + combined_text
        )
        final_uuid = save_doc_hash(site=site, url=None, kind="final", summary=final_summary)

        return {
            "pages": page_entries,
            "final_summary": {"uuid": final_uuid, "summary": final_summary},
        }


# ========= FASTAPI ROUTES =========
@app.get("/crawl")
async def crawl(
    url: str = Query(..., description="Target URL untuk crawling"),
    depth: int = Query(2, description="Maximum depth untuk crawling"),
    pages: int = Query(5, description="Maximum number of pages untuk crawling"),
):
    try:
        result = await crawl_and_analyze(url, depth, pages)
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)


@app.get("/search")
def search(
    q: str = Query(..., description="Query text untuk semantic search"),
    site: Optional[str] = Query(None, description="Filter site (opsional)"),
    k: int = Query(5, description="Top-K"),
):
    try:
        ensure_hash_index()
        hits = semantic_search(q, top_k=k, site=site)
        return JSONResponse(content={"success": True, "results": hits}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)


@app.get("/chat")
def chat(
    q: str = Query(..., description="Pertanyaan user"),
    site: Optional[str] = Query(None, description="Filter site (opsional)"),
    k: int = Query(5, description="Jumlah konteks"),
):
    """
    Jawaban berbasis konteks dari Redis (RAG sederhana).
    """
    try:
        ensure_hash_index()
        hits = semantic_search(q, top_k=k, site=site)

        context_blocks = []
        for h in hits:
            block = f"- [kind:{h['kind']}] {h['url']}\n{h['summary']}"
            context_blocks.append(block)
        context = "\n\n".join(context_blocks)

        prompt = (
            "You are a helpful assistant. Use ONLY the context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {q}\n\n"
            "Answer in concise Indonesian. If the answer is not in context, say you don't know."
        )
        answer = gemini_request(prompt)

        return JSONResponse(
            content={"success": True, "answer": answer, "sources": hits}, status_code=200
        )
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)
