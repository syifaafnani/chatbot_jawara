import os, io, math, json, hashlib, pathlib
from dataclasses import dataclass
from typing import List, Dict, Tuple
from dotenv import load_dotenv

import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import faiss
import tiktoken

from openai import OpenAI

# =========================
# Konfigurasi
# =========================
load_dotenv()
client = OpenAI()

EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL   = "gpt-4o-mini"                 # teks & vision
CHUNK_TOKENS = 400                          # kira-kira ~300-500 kata
CHUNK_OVERLAP_TOKENS = 60
TOP_K = 3
DATA_DIR = "docs"                           # folder input PDF
STORE_DIR = "vectorstore"                   # output vektor & metadata
SESSION_ID = "session_id"                   # sesi user
os.makedirs(STORE_DIR, exist_ok=True)

# =========================
# Util
# =========================
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    # gunakan encoding cl100k_base untuk pendekatan
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def chunk_text(text: str,
               max_tokens: int = CHUNK_TOKENS,
               overlap_tokens: int = CHUNK_OVERLAP_TOKENS) -> List[str]:
    """
    Simple recursive chunking by sentences with token budget & overlap.
    """
    # split based on sentence-ish delimiters
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    enc = tiktoken.get_encoding("cl100k_base")

    chunks = []
    cur = []
    cur_tokens = 0

    for sent in sentences:
        stoks = enc.encode(sent)
        if cur_tokens + len(stoks) > max_tokens and cur:
            chunks.append(" ".join(cur).strip())
            # create overlap
            if overlap_tokens > 0:
                overlap = []
                # take tail of current chunk tokens
                toks = enc.encode(chunks[-1])
                # map back to text roughly
                tail = enc.decode(toks[-overlap_tokens:]) if len(toks) > overlap_tokens else chunks[-1]
                overlap.append(tail)
                cur = overlap.copy()
                cur_tokens = len(enc.encode(" ".join(cur)))
            else:
                cur = []
                cur_tokens = 0
        cur.append(sent)
        cur_tokens += len(stoks)

    if cur:
        chunks.append(" ".join(cur).strip())

    # remove empties & very short
    chunks = [c for c in chunks if c and len(c.split()) > 3]
    return chunks

# =========================
# Ekstraksi dari PDF
# =========================
@dataclass
class PageContent:
    doc_path: str
    page_num: int  # 1-based
    text: str
    vision_caption: str = ""  # hasil captioning untuk halaman minim teks (opsional)

def extract_pdf(filepath: str,
                vision_caption_if_low_text: bool = True,
                min_text_chars: int = 200) -> List[PageContent]:
    """
    Ekstrak teks per halaman; jika teks sangat minim dan opsi vision aktif,
    render halaman jadi PNG dan minta caption/ringkasan ke GPT-4o-mini (vision).
    """
    doc = fitz.open(filepath)
    pages: List[PageContent] = []

    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text").strip()

        vision_caption = ""
        if vision_caption_if_low_text and len(text) < min_text_chars:
            # render ke image in-memory (PNG)
            pix = page.get_pixmap(dpi=180, alpha=False)
            img_bytes = pix.tobytes("png")

            # kirim ke GPT-4o-mini untuk minta deskripsi tabel/gambar
            try:
                msg = [
                    {"role": "system", "content": "Anda adalah asisten yang mendeskripsikan isi halaman PDF (grafik/tabel/gambar) menjadi teks informatif untuk keperluan RAG."},
                    {"role": "user", "content": [
                        {"type": "text", "text": (
                            "Ringkas isi visual halaman ini. Jika ada tabel atau angka penting, "
                            "tuliskan sebagai bullet ringkas. Output bahasa Indonesia."
                        )},
                        {"type": "image", "image_data": img_bytes}
                    ]}
                ]
                resp = client.chat.completions.create(
                    model=GEN_MODEL,
                    messages=msg,
                    temperature=0.2
                )
                vision_caption = resp.choices[0].message.content.strip()
            except Exception as e:
                vision_caption = f"(Gagal caption vision: {e})"

        pages.append(PageContent(
            doc_path=filepath,
            page_num=i+1,
            text=text,
            vision_caption=vision_caption
        ))

    doc.close()
    return pages

# =========================
# Embedding & Vector Store
# =========================
def embed_texts(texts: List[str]) -> np.ndarray:
    # batched embeddings
    BATCH = 128
    vectors = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch
        )
        batch_vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
        vectors.extend(batch_vecs)
    return np.vstack(vectors)

def build_or_update_index(pdfs_dir: str = DATA_DIR, store_dir: str = STORE_DIR):
    """
    Baca semua PDF di folder, ekstrak, chunk, embedding, simpan ke FAISS + Parquet.
    Jika sudah ada store, akan *append* dokumen baru (berbasis hash path+mtime).
    """
    pathlib.Path(store_dir).mkdir(parents=True, exist_ok=True)
    meta_path = os.path.join(store_dir, "chunks.parquet")
    index_path = os.path.join(store_dir, "index.faiss")

    # muat metadata lama jika ada
    if os.path.exists(meta_path):
        df = pd.read_parquet(meta_path)
    else:
        df = pd.DataFrame(columns=["doc_id","source","page","chunk","chunk_id"])

    # dokumen yang sudah ada (berdasar doc_id)
    known = set(df["doc_id"].unique()) if len(df) else set()

    new_rows = []
    for name in os.listdir(pdfs_dir):
        if not name.lower().endswith(".pdf"): continue
        fpath = os.path.join(pdfs_dir, name)
        # doc_id unik dari path + mtime
        stat = os.stat(fpath)
        doc_id = sha1(f"{os.path.abspath(fpath)}::{int(stat.st_mtime)}")

        if doc_id in known:
            print(f"Skip (sudah terindeks): {name}")
            continue

        print(f"Ekstrak: {name}")
        pages = extract_pdf(fpath)

        for p in pages:
            base = p.text
            if p.vision_caption and len(p.vision_caption) > 10:
                # gabungkan caption agar informasi visual ikut terindeks
                base = (base + "\n\n[Deskripsi Visual]\n" + p.vision_caption).strip()

            if not base or len(base.split()) < 5:
                continue

            # chunking
            chunks = chunk_text(base, CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS)
            for idx, ch in enumerate(chunks):
                new_rows.append({
                    "doc_id": doc_id,
                    "source": name,
                    "page": p.page_num,
                    "chunk": ch,
                    "chunk_id": f"{doc_id}-{p.page_num}-{idx}"
                })

    if not new_rows:
        print("Tidak ada dokumen baru untuk diindeks.")
        return

    add_df = pd.DataFrame(new_rows)
    # embed
    print(f"Embedding {len(add_df)} chunk ...")
    vecs = embed_texts(add_df["chunk"].tolist())

    # Normalisasi ke unit length untuk Cosine similarity via Inner Product
    faiss.normalize_L2(vecs)

    # gabung dengan yang lama
    if os.path.exists(index_path) and len(df):
        # load index lama
        old_vecs = None
        index = faiss.read_index(index_path)
        # FAISS tidak menyimpan vecs lama terpisah, jadi kita tinggal add vecs baru ke index
        index.add(vecs)
        faiss.write_index(index, index_path)
        full_df = pd.concat([df, add_df], ignore_index=True)
        full_df.to_parquet(meta_path, index=False)
    else:
        # buat index baru
        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product for cosine (setelah normalisasi)
        index.add(vecs)
        faiss.write_index(index, index_path)
        add_df.to_parquet(meta_path, index=False)

    print(f"Selesai. Tersimpan ke: {index_path} & {meta_path}")

def load_store(store_dir: str = STORE_DIR) -> Tuple[faiss.Index, pd.DataFrame]:
    index_path = os.path.join(store_dir, "index.faiss")
    meta_path  = os.path.join(store_dir, "chunks.parquet")
    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        raise FileNotFoundError("Vector store belum dibuat. Jalankan build_or_update_index() dulu.")
    index = faiss.read_index(index_path)
    df = pd.read_parquet(meta_path)
    return index, df

# =========================
# Retrieval & Jawaban
# =========================
def retrieve(query: str, top_k: int = TOP_K, store_dir: str = STORE_DIR) -> List[Dict]:
    index, df = load_store(store_dir)
    # embed query
    qvec = embed_texts([query]).astype(np.float32)
    faiss.normalize_L2(qvec)

    D, I = index.search(qvec, top_k)   # cosine via inner product
    I = I[0]
    D = D[0]
    res = []
    for score, idx in zip(D, I):
        if idx < 0: continue
        row = df.iloc[idx].to_dict()
        row["score"] = float(score)
        res.append(row)
    return res

def answer(query: str, sessionID: str = SESSION_ID, top_k: int = TOP_K, store_dir: str = STORE_DIR) -> Dict:
    hits = retrieve(query, top_k, store_dir)
    if not hits:
        return {"answer": "Maaf, tidak ada konteks yang cocok di indeks.", "citations": []}

    # Compose context
    context_blocks = []
    for h in hits:
        header = f"[{h['source']} p.{h['page']}]"
        context_blocks.append(f"{header}\n{h['chunk']}")
    context = "\n\n---\n\n".join(context_blocks)

    sys = (
        "Anda asisten QA yang menjawab pertanyaan terkait Kesesuaian Kegiatan Pemanfaatan Ruang Laut (KKPRL)."
        "Hanya jawab dari KONTEN KONTEKS yang diberikan, dahulukan dari file Buku Saku Panduan dan Persyaratan Dasar dari OSS. "
        "Jika jawabannya tidak ada di konteks, sampaikan maaf karena tidak dapat menemukan informasi tersebut. "
        "Sertakan sitasi [nama_file p.halaman] pada kalimat yang relevan."
        "Sertakan link (tanpa ditutup kurung) yang terdapat dalam dokumen jika ada pertanyaan yang relevan."
    )
    user_prompt = (
        f"PERTANYAAN:\n{query}\n\n"
        f"KONTEN KONTEKS (kutipan hasil pencarian):\n{context}\n\n"
        "Buat jawaban ringkas, jelas, dan faktual. Gunakan bahasa yang mudah dimengerti orang awam dan bullet seperlunya."
    )

    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role":"system","content":sys},
            {"role":"user","content":user_prompt}
        ],
        temperature=0.2
    )
    content = resp.choices[0].message.content.strip()

    # siapkan sitasi terstruktur
    citations = [
        {"source": h["source"], "page": int(h["page"]), "score": round(h["score"], 4)}
        for h in hits
    ]
    return {"sessionID": sessionID, "answer": content, "citations": citations}

# =========================
# CLI sederhana
# =========================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="RAG PDF: index & tanya-jawab")
    ap.add_argument("--build", action="store_true", help="Bangun/Update indeks dari folder pdfs/")
    ap.add_argument("--ask", type=str, help="Ajukan pertanyaan yang akan dijawab berdasarkan indeks")
    ap.add_argument("--pdfs", type=str, default=DATA_DIR, help="Folder PDF input")
    ap.add_argument("--store", type=str, default=STORE_DIR, help="Folder penyimpanan indeks")
    ap.add_argument("--topk", type=int, default=TOP_K, help="Jumlah potongan yang diambil")
    args = ap.parse_args()

    if args.build:
        build_or_update_index(args.pdfs, args.store)

    if args.ask:
        res = answer(args.ask, args.topk, args.store)
        print("\n=== JAWABAN ===\n")
        print(res["answer"])
        print("\n=== SITASI ===")
        for c in res["citations"]:
            print(f"- {c['source']} p.{c['page']} (score={c['score']})")
