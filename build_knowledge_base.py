

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import re
import shutil
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
KB_DIR = PROJECT_DIR / "data" / "knowledge_base"

RAW_DIR = KB_DIR / "raw"
PROCESSED_DIR = KB_DIR / "processed"
CHUNKS_DIR = KB_DIR / "chunks"
INDEX_DIR = KB_DIR / "index"

FAISS_INDEX_PATH = INDEX_DIR / "knowledge_faiss_clean.index"
METADATA_PATH = INDEX_DIR / "knowledge_metadata_clean.json"

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

MAX_CHUNK_WORDS = 180

# End-user documents that complement the existing admin-level guides.
# KB-NETWORK-* documents are general network troubleshooting and are
# shared by the VPN, DNS, DHCP and Wireless categories (see rag_engine.py).
NEW_DOCUMENTS = [
    {
        "document_id": "KB-WIFI-002",
        "url": (
            "https://support.microsoft.com/en-us/windows/"
            "fix-wi-fi-connection-issues-in-windows-"
            "9424a1f7-6a3b-65a6-4d78-7f07eee84d2c"
        ),
    },
    {
        "document_id": "KB-VPN-002",
        "url": (
            "https://support.microsoft.com/en-us/windows/"
            "connect-to-a-vpn-in-windows-"
            "3d29aeb1-f497-f6b7-7633-115722c1009c"
        ),
    },
    {
        "document_id": "KB-EMAIL-002",
        "url": (
            "https://support.microsoft.com/en-us/office/"
            "i-can-t-send-or-receive-messages-in-outlook-"
            "97748418-bbd5-4743-a05b-581f22a466dd"
        ),
    },
    {
        "document_id": "KB-EMAIL-003",
        "url": (
            "https://learn.microsoft.com/en-us/troubleshoot/exchange/"
            "outlook-issues/outlook-connection-issues"
        ),
    },
    {
        "document_id": "KB-NETWORK-001",
        "url": (
            "https://support.microsoft.com/en-us/windows/"
            "fix-ethernet-connection-problems-in-windows-"
            "2311254e-cab8-42d6-90f3-cb0b9f63645f"
        ),
    },
]

# Sections that end the article (page footer).
STOP_HEADINGS = {
    "feedback",
    "additional resources",
    "related content",
    "related topics",
    "need more help?",
    "want more options?",
    "was this information helpful?",
    "subscribe rss feeds",
}

SKIP_HEADINGS = {
    "in this article",
}

# Website text that is not troubleshooting content.
NOISE_PATTERNS = [
    r"^note$",
    r"^tip$",
    r"^important$",
    r"^windows 1[01]$",
    r"^access to this page requires authorization",
    r"^try our virtual agent",
    r"^applies to",
    r"^run the troubleshooter in get help$",
]

EXCLUDED_PARENTS = [
    "nav",
    "footer",
    "aside",
]


# ============================================================
# 2. DOWNLOAD
# ============================================================

def download_document(document, force=False):
    """
    Download the HTML page once and keep it in data/knowledge_base/raw.
    """

    raw_path = RAW_DIR / f"{document['document_id']}.html"

    if raw_path.exists() and not force:
        print(f"  Using saved HTML: {raw_path.name}")
        return raw_path.read_text(encoding="utf-8")

    response = requests.get(
        document["url"],
        timeout=30,
        headers={"User-Agent": "EnterpriseSupportCopilot/1.0"},
    )

    response.raise_for_status()

    raw_path.write_text(response.text, encoding="utf-8")

    print(f"  Downloaded {len(response.text):,} characters")

    return response.text


# ============================================================
# 3. EXTRACT HEADINGS AND TEXT
# ============================================================

def clean_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    # get_text(" ") adds a space before punctuation after inline tags
    return re.sub(r"\s+([.,;:!?)])", r"\1", text)


def is_noise(text):
    lowered = text.lower()

    return any(
        re.search(pattern, lowered)
        for pattern in NOISE_PATTERNS
    )


def extract_blocks(html):
    """
    Return the article as an ordered list of (tag, text) blocks.

    Only the outermost p / li / pre / table elements are kept,
    so nested text is not duplicated.
    """

    soup = BeautifulSoup(html, "html.parser")

    for element in soup(["script", "style", "noscript", "svg"]):
        element.decompose()

    root = soup.find("main") or soup.body

    if root is None:
        raise ValueError("No article content found.")

    blocks = []
    block_tags = {"p", "li", "pre", "table"}

    for element in root.find_all(
        ["h1", "h2", "h3", "h4", "p", "li", "pre", "table"]
    ):

        if element.find_parent(EXCLUDED_PARENTS):
            continue

        if element.find_parent(attrs={"role": "navigation"}):
            continue

        if (
            element.name in block_tags
            and element.find_parent(list(block_tags))
        ):
            continue

        separator = "\n" if element.name == "pre" else " "

        text = element.get_text(separator, strip=True)

        if element.name != "pre":
            text = clean_text(text)

        if text:
            blocks.append((element.name, text))

    return blocks


# ============================================================
# 4. CHUNKING
# ============================================================

def split_long_text(text, max_words=MAX_CHUNK_WORDS):
    """
    Split long sections into parts of about max_words.
    Lines (list items, commands) are kept whole; a single very long
    line is split at sentence boundaries.
    """

    pieces = []

    for line in text.split("\n"):

        if len(line.split()) <= max_words:
            pieces.append(line)
        else:
            pieces.extend(re.split(r"(?<=[.!?])\s+", line))

    parts = []
    current = []
    current_words = 0

    for piece in pieces:

        words = len(piece.split())

        if current and current_words + words > max_words:
            parts.append("\n".join(current))
            current = []
            current_words = 0

        current.append(piece)
        current_words += words

    if current:
        parts.append("\n".join(current))

    return parts


def build_chunks(document, blocks):
    """
    One chunk per heading section, with the full heading path
    (document title > section > subsection).
    """

    document_id = document["document_id"]

    sections = []
    heading_path = {}
    current = None
    inside_article = False
    document_title = None

    for tag, text in blocks:

        if tag in {"h1", "h2", "h3", "h4"}:

            level = int(tag[1])

            if tag == "h1" and not inside_article:
                inside_article = True
                document_title = text

            if not inside_article:
                continue

            if text.lower() in STOP_HEADINGS:
                break

            if text.lower() in SKIP_HEADINGS:
                continue

            if current and current["lines"]:
                sections.append(current)

            heading_path[level] = text

            for previous_level in list(heading_path):
                if previous_level > level:
                    del heading_path[previous_level]

            current = {
                "heading": text,
                "heading_path": [
                    heading_path[key]
                    for key in sorted(heading_path)
                ],
                "lines": [],
            }

        elif inside_article and current is not None:

            if is_noise(text):
                continue

            prefix = "- " if tag == "li" else ""
            current["lines"].append(prefix + text)

    if current and current["lines"]:
        sections.append(current)

    if not sections:
        raise ValueError(
            f"{document_id}: no sections extracted. "
            "The page layout may have changed."
        )

    # Microsoft Support pages repeat sections for Windows 10 / 11.
    # Keep only the first copy of identical sections.
    seen = set()
    unique_sections = []

    for section in sections:

        content = "\n".join(section["lines"])
        key = (section["heading"], content)

        if key in seen:
            continue

        seen.add(key)
        section["content"] = content
        unique_sections.append(section)

    downloaded_at = datetime.now(timezone.utc).isoformat()

    chunks = []

    for section in unique_sections:

        for part in split_long_text(section["content"]):

            chunks.append({
                "document_id": document_id,
                "chunk_id": f"{document_id}-{len(chunks) + 1:03d}",
                "document_title": document_title,
                "heading": section["heading"],
                "heading_path": deepcopy(section["heading_path"]),
                "content": part,
                "source_url": document["url"],
                "publisher": "Microsoft",
                "downloaded_at": downloaded_at,
                "language": "en",
            })

    return document_title, chunks


def save_document(document, title, chunks):

    document_id = document["document_id"]

    processed_text = "\n\n".join(
        f"{' > '.join(chunk['heading_path'])}\n{chunk['content']}"
        for chunk in chunks
    )

    (PROCESSED_DIR / f"{document_id}.txt").write_text(
        processed_text,
        encoding="utf-8",
    )

    (CHUNKS_DIR / f"{document_id}.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    words = sum(len(chunk["content"].split()) for chunk in chunks)

    print(f"  Title: {title}")
    print(f"  Chunks: {len(chunks)} | Words: {words:,}")


# ============================================================
# 5. CLEAN CORPUS (same rules as rag_pipeline.ipynb)
# ============================================================

def is_useful_chunk(chunk):

    text = chunk["content"].strip()
    heading = chunk["heading"].strip().lower()

    excluded_headings = {
        "known issues",
        "more information",
        "feedback",
        "additional resources",
    }

    if heading in excluded_headings:
        return False

    technical_patterns = [
        r"\berror\b",
        r"0x[0-9a-f]+",
        r"\bipconfig\b",
        r"\bnslookup\b",
        r"\bcertificate\b",
        r"\bIKE\b",
    ]

    if any(
        re.search(pattern, heading + " " + text, re.IGNORECASE)
        for pattern in technical_patterns
    ):
        return True

    return len(text.split()) >= 15


# ============================================================
# 6. REBUILD THE INDEX
# ============================================================

def rebuild_index():

    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    all_chunks = []

    for path in sorted(CHUNKS_DIR.glob("KB-*.json")):
        all_chunks.extend(
            json.loads(path.read_text(encoding="utf-8"))
        )

    chunk_ids = [chunk["chunk_id"] for chunk in all_chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs found.")

    clean_chunks = [
        chunk for chunk in all_chunks
        if is_useful_chunk(chunk)
    ]

    print(f"\nAll chunks: {len(all_chunks)}")
    print(f"Clean chunks: {len(clean_chunks)}")

    # Back up the current index before replacing it.
    backup_dir = INDEX_DIR / (
        "backup_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    backup_dir.mkdir(parents=True, exist_ok=True)

    for path in [FAISS_INDEX_PATH, METADATA_PATH]:
        if path.exists():
            shutil.copy2(path, backup_dir / path.name)

    print(f"Old index backed up to: {backup_dir.relative_to(PROJECT_DIR)}")

    embedding_texts = [
        f"Topic: {' > '.join(chunk['heading_path'])}\n"
        f"Content: {chunk['content']}"
        for chunk in clean_chunks
    ]

    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    embeddings = model.encode(
        embedding_texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    assert np.isfinite(embeddings).all(), "Invalid embedding values."

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_INDEX_PATH))

    METADATA_PATH.write_text(
        json.dumps(clean_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nIndex rebuilt successfully!")
    print("Vectors:", index.ntotal)

    counts = {}

    for chunk in clean_chunks:
        counts[chunk["document_id"]] = counts.get(chunk["document_id"], 0) + 1

    print("\nChunks per document:")

    for document_id, count in sorted(counts.items()):
        print(f"  {document_id:<16} {count}")


# ============================================================
# 7. MAIN
# ============================================================

def main():

    force_download = "--force-download" in sys.argv
    index_only = "--index-only" in sys.argv

    for directory in [RAW_DIR, PROCESSED_DIR, CHUNKS_DIR, INDEX_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    if not index_only:

        for document in NEW_DOCUMENTS:

            print(f"\n{document['document_id']}")

            html = download_document(document, force=force_download)
            blocks = extract_blocks(html)
            title, chunks = build_chunks(document, blocks)
            save_document(document, title, chunks)

    rebuild_index()


if __name__ == "__main__":
    main()
