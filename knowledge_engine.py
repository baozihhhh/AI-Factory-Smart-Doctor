import os
import json
import hashlib
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import RAW_DOCS_DIR, FAISS_INDEX_PATH, CHUNKS_META_PATH, VECTOR_INDEX_DIR

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DOCS_HASH_PATH = os.path.join(VECTOR_INDEX_DIR, "docs_hash.json")

# ── 本地嵌入模型（单例） ──
_embedding_model = None


def _get_model() -> SentenceTransformer:
    """懒加载本地 sentence-transformers 模型（单例）。"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def extract_text(file_path: str) -> str:
    """从 PDF 或 TXT 文件提取文本。"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        import PyPDF2
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[str]:
    """将长文本切分为重叠的小段。"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


# ── 本地嵌入（替代远程 API） ──

def get_embedding(text: str) -> list[float]:
    """使用本地模型生成单条文本的嵌入向量。"""
    model = _get_model()
    emb = model.encode(text, normalize_embeddings=True)
    return emb.tolist()


def get_embeddings_batch(texts: list[str], batch_size: int = 64,
                         progress_callback=None) -> np.ndarray:
    """批量生成嵌入向量，支持进度回调。返回 float32 numpy 矩阵。"""
    model = _get_model()
    all_embs = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        embs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embs.append(embs)
        if progress_callback:
            done = min(i + batch_size, total)
            pct = 0.3 + 0.6 * (done / total)
            progress_callback(pct, f"正在生成向量嵌入... ({done}/{total})")
    return np.vstack(all_embs).astype("float32")


# ── 缓存指纹 ──

def _compute_docs_hash() -> str:
    """根据 raw_docs 目录下所有文件的名称+大小+修改时间生成哈希指纹。"""
    docs = list_uploaded_docs()
    if not docs:
        return ""
    parts = []
    for name in sorted(docs):
        path = os.path.join(RAW_DOCS_DIR, name)
        stat = os.stat(path)
        parts.append(f"{name}|{stat.st_size}|{stat.st_mtime}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def _load_cached_hash() -> str:
    """读取上次构建索引时保存的文档指纹。"""
    if not os.path.exists(DOCS_HASH_PATH):
        return ""
    try:
        with open(DOCS_HASH_PATH, "r") as f:
            return json.load(f).get("hash", "")
    except Exception:
        return ""


def _save_docs_hash(h: str):
    """保存文档指纹。"""
    os.makedirs(VECTOR_INDEX_DIR, exist_ok=True)
    with open(DOCS_HASH_PATH, "w") as f:
        json.dump({"hash": h}, f)


def _index_is_fresh() -> bool:
    """检查本地索引是否与当前文档一致（Cache First）。"""
    if not has_index():
        return False
    return _compute_docs_hash() == _load_cached_hash()


# ── 公共接口 ──

def list_uploaded_docs() -> list[str]:
    """列出已上传的原始文档。"""
    if not os.path.exists(RAW_DOCS_DIR):
        return []
    return [f for f in os.listdir(RAW_DOCS_DIR)
            if f.lower().endswith((".pdf", ".txt"))]


def build_index(client=None, progress_callback=None) -> int:
    """扫描 raw_docs 目录，构建 FAISS 索引。返回总 chunk 数。

    如果文档未变动且索引已存在，直接复用缓存（不重新生成嵌入）。
    client 参数保留以兼容旧调用，但不再使用。
    """
    docs = list_uploaded_docs()
    if not docs:
        return 0

    # Cache First：文档未变动则直接返回已有索引的 chunk 数
    if _index_is_fresh():
        try:
            with open(CHUNKS_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if progress_callback:
                progress_callback(1.0, "索引未变动，已直接加载缓存！")
            return len(meta)
        except Exception:
            pass  # 缓存损坏，继续重建

    if progress_callback:
        progress_callback(0.1, "正在解析文档并分块...")

    all_chunks = []
    meta = []

    for doc_name in docs:
        path = os.path.join(RAW_DOCS_DIR, doc_name)
        try:
            text = extract_text(path)
        except Exception as e:
            if progress_callback:
                progress_callback(0.1, f"跳过文件 {doc_name}: {e}")
            continue
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            meta.append({"doc": doc_name, "chunk_idx": i, "text": chunk})

    if not all_chunks:
        return 0

    if progress_callback:
        progress_callback(0.3, "正在生成向量嵌入...")

    try:
        emb_matrix = get_embeddings_batch(all_chunks, progress_callback=progress_callback)
    except Exception as e:
        raise RuntimeError(
            f"嵌入生成失败: {e}\n请检查 sentence-transformers 是否已正确安装。"
        )

    dim = emb_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_matrix)

    os.makedirs(VECTOR_INDEX_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CHUNKS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    # 保存文档指纹用于下次缓存校验
    _save_docs_hash(_compute_docs_hash())

    if progress_callback:
        progress_callback(1.0, "索引构建完成！")

    return len(all_chunks)


def search_knowledge(client=None, query: str = "", top_k: int = 3) -> list[dict]:
    """在知识库中检索与 query 最相关的 top_k 个片段。

    client 参数保留以兼容旧调用，但不再使用。
    """
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(CHUNKS_META_PATH):
        return []

    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(CHUNKS_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)

        query_emb = np.array([get_embedding(query)], dtype="float32")

        scores, indices = index.search(query_emb, min(top_k, index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = meta[idx].copy()
            entry["score"] = float(score)
            results.append(entry)
        return results
    except Exception:
        return []


def save_uploaded_file(uploaded_file) -> str:
    """保存 Streamlit UploadedFile 到 raw_docs 目录，返回文件路径。"""
    os.makedirs(RAW_DOCS_DIR, exist_ok=True)
    dest = os.path.join(RAW_DOCS_DIR, uploaded_file.name)
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


def delete_doc(filename: str):
    """删除一个已上传的文档。"""
    path = os.path.join(RAW_DOCS_DIR, filename)
    if os.path.exists(path):
        os.remove(path)


def has_index() -> bool:
    return os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_META_PATH)
