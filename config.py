import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 工程师知识库
RAW_DOCS_DIR = os.path.join(DATA_DIR, "raw_docs")
VECTOR_INDEX_DIR = os.path.join(DATA_DIR, "vector_index")
FAISS_INDEX_PATH = os.path.join(VECTOR_INDEX_DIR, "index.faiss")
CHUNKS_META_PATH = os.path.join(VECTOR_INDEX_DIR, "chunks.json")

# 诊断历史持久化
HISTORY_FILE_PATH = os.path.join(DATA_DIR, "diagnosis_history.json")
HISTORY_MAX_RECORDS = 200

# PDF 资源目录
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
FONTS_DIR = os.path.join(ASSETS_DIR, "fonts")

# 确保目录存在
for d in [DATA_DIR, RAW_DOCS_DIR, VECTOR_INDEX_DIR, ASSETS_DIR, FONTS_DIR]:
    os.makedirs(d, exist_ok=True)
