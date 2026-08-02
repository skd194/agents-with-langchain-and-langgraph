# 4. Document Loading & Chunking

> **Quick recall:** LLMs need context, and context comes from your data. **Document loaders** turn raw files (PDF, TXT, HTML, DOCX, CSV, web pages) into a standard `Document` object — `page_content` + `metadata` — that's easy to work with. Then **chunking** splits those documents into smaller pieces for embedding.

This is the first step of the RAG indexing pipeline → see [05-rag-and-embeddings.md](05-rag-and-embeddings.md).

---

## Raw files → `Document` objects

```
PDF / TXT / HTML / DOCX / CSV ──▶ loader.load() ──▶ [Document, Document, ...]
        (raw files)                              (code-friendly, structured)
```

A loader returns a **list of `Document`s**, each with:

```
Document
├─ page_content   "the actual extracted text..."   (required, str)
└─ metadata       {"source": "report.pdf", "page": 3, "author": "...", ...}  (any keys)
```

### Building / inspecting a `Document` (`006_document_loaders.py`, `doc_structure`)

You can construct one by hand — only `page_content` is required; `metadata` takes any keys you want.

```python
from langchain_core.documents import Document

doc = Document(
    page_content="This is a sample document.",   # required, str
    metadata={                                    # arbitrary fields
        "source": "manual_creation.txt",
        "author": "Author",
        "length": 30,
        "tags": ["sample", "test"],
        "created_at": "2024-06-01",
    },
)
print(type(doc.page_content))   # <class 'str'>
print(doc.metadata["author"])   # Author
```

**Updating** — documents are effectively immutable, so make a **new** one, spreading the old fields:

```python
updated = Document(
    page_content=doc.page_content + " More content.",
    metadata={**doc.metadata, "updated": True},   # spread old + add/override
)
```

## Core loaders

| Loader | Loads | Notes |
|--------|-------|-------|
| `PyPDFLoader` | PDF files | Fast, basic extraction |
| `TextLoader` | Plain `.txt` | Simplest |
| `DirectoryLoader` | A folder of many files | Point at a dir, load in bulk |
| `WebBaseLoader` | Web pages | Pass a URL (or list of URLs) |
| `UnstructuredLoader` | Complex/mixed formats (MD, DOCX, ...) | Handles messy layouts |

> Loaders live in `langchain_community` → install with `uv add langchain-community` (also brings text splitters and other modules).

**Basic pattern** — instantiate with the source, then `.load()`:

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("report.pdf")   # instantiate with source
docs   = loader.load()               # -> list[Document]
```

### TextLoader in code (`006_document_loaders.py`)

```python
from langchain_community.document_loaders import TextLoader

loader    = TextLoader(file_path)     # a .txt path
documents = loader.load()             # -> list[Document]

print(len(documents))                        # e.g. 1
print(documents[0].page_content[:100])       # the extracted text
print(documents[0].metadata)                 # {'source': '...path...'}
```

- Even a simple text file comes back as a **list** of `Document`s (here just one).
- Access the text with `.page_content` and the auto-added `.metadata` (`source`, and more depending on the loader).
- The value of loaders: they attach useful **metadata** you'd otherwise have to build yourself.

## Choosing a PDF loader

| Loader | Speed | Metadata | Best for |
|--------|-------|----------|----------|
| `PyPDFLoader` | Fast | Basic | Simple PDFs (recommended starting point) |
| `PyMuPDFLoader` | **Fastest** | Good | High volume; strong all-rounder |
| `UnstructuredPDFLoader` | Slower | **Detailed** | Complex layouts & **tables** |

> Start with `PyPDFLoader`; switch to another later based on your use case (e.g. tables → `UnstructuredPDFLoader`).

### PyPDFLoader in code (`006_document_loaders.py`, `pdf_loader`)

Needs the `pypdf` backend → `uv add pypdf`.

```python
from langchain_community.document_loaders import PyPDFLoader

loader    = PyPDFLoader("docs/langchain_demo.pdf")
documents = loader.load()                      # -> one Document per PAGE

print(len(documents))                          # e.g. 3 (a 3-page PDF)
for doc in documents:
    print(doc.page_content[:100])
    print(doc.metadata)
```

- A multi-page PDF loads as a **list of `Document`s — one per page** (not one doc for the whole file).
- `pypdf` auto-fills rich metadata: `source`, `total_pages`, `page`, `page_label`, plus `producer` / `creator` / `creationdate` pulled from the PDF itself.

## Web loading

Scrape web pages into `Document` objects (`006_document_loaders.py`, `web_loader`).

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    "https://en.wikipedia.org/wiki/Web_scraping",
    bs_kwargs={"parse_only": None},        # BeautifulSoup options (None = parse everything)
)
docs = loader.load()

print(docs[0].metadata.get("source"))      # the URL
print(len(docs[0].page_content))           # extracted text length
print(docs[0].page_content[:200])          # preview

# multiple URLs → list of Documents (one+ per page)
docs = WebBaseLoader(["https://example.com/1", "https://example.com/2"]).load()
```

- **Requires BeautifulSoup** → `uv add bs4` (imported as `from bs4 import BeautifulSoup`). Without it, `.load()` errors.
- **`bs_kwargs`** passes options to BeautifulSoup, e.g. `parse_only` (a `SoupStrainer` to keep only certain tags like `div`; `None` = everything) and `features="html.parser"`.
- Other tunable params: `proxies`, `verify_ssl`, `header_template`, `encoding`, `requests_per_second`.
- Metadata includes the `source` URL.

## Directory loading

Load many files at once by pointing at a folder:

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="docs/",
    glob="**/*.pdf",          # which files to match
    loader_cls=PyPDFLoader,   # loader to use per file
)
docs = loader.load()
```

- **`glob` = pattern filter.** `**/*.pdf` = every `.pdf` in `docs/` **and all subdirectories** (`**` = recurse, `*` = any name).
- `loader_cls` tells `DirectoryLoader` which loader to apply to each matched file.

### Lazy loading large file sets (`006_document_loaders.py`, `lazy_loader`)

For big directories, `.lazy_load()` yields documents **one at a time** instead of building the whole list in memory.

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(tmpdir, glob="*.txt", loader_cls=TextLoader)

for doc in loader.lazy_load():             # generator — memory-efficient
    print(doc.page_content[:50])
    print(doc.metadata["source"])
```

- `.load()` returns the full `list[Document]` at once; `.lazy_load()` returns a **generator** you iterate.
- Prefer `.lazy_load()` for large data sets → lower peak memory (process each doc, then discard).

---

## Chunking

Loaded documents are often too big to embed or feed to an LLM, so **text splitters** cut them into smaller chunks before embedding.

```
[big Document ~50k tokens] ──▶ text splitter ──▶ [chunk, chunk, ...]  (~500–1000 tokens each)
                                                        │ embedding model
                                                        ▼
                                                   vectors → vector DB
```

**Why it matters:** LLMs have context limits, so chunks must fit the token window. Chunk size is a balance:
- **Too small** → loses context (incomplete info).
- **Too large** → noisy retrieval (pulls in irrelevant content).
- **Just right (middle)** → precise retrieval with good relevance.

### RecursiveCharacterTextSplitter (default)

The go-to splitter (`007_text_splitters.py`, `recursive_splitter`). Splits on a **priority of separators**, backing off only when needed: **paragraphs (`\n\n`) → lines (`\n`) → spaces → characters**. This preserves **semantic coherence** — it keeps paragraphs whole first, then sentences, then words, so text isn't cut mid-thought.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # max length per chunk (see unit note below)
    chunk_overlap=50,     # length shared between adjacent chunks
    separators=["\n\n", "\n", " ", ""],   # where to cut, in priority order
)

chunks = splitter.split_text(sample_text)          # raw string  -> list[str]
chunks = splitter.split_documents(documents)       # Document(s) -> list[Document] (keeps metadata)
```

- **`split_text` vs `split_documents`:** use `split_text` for a plain string, `split_documents` for loaded `Document` objects (it carries metadata through to each chunk).

- **Unit = characters by default.** `chunk_size` / `chunk_overlap` are measured by the splitter's `length_function`, which defaults to `len` (character count). So `chunk_size=1000` ≈ 1000 characters.
- **Count in tokens instead** with the tiktoken factory — then the numbers mean tokens:
  ```python
  splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
      chunk_size=1000, chunk_overlap=200)   # now 1000 = tokens
  ```
- **`chunk_overlap`** repeats a slice of the end of one chunk at the start of the next → preserves context across boundaries. Bigger overlap = more context retained, but more redundancy — keep it balanced.

```
chunk1 ▓▓▓▓▓▓▓░░░           ░░░ = overlap shared
chunk2       ░░░▓▓▓▓▓▓▓░░░       with the neighbour
chunk3             ░░░▓▓▓▓▓▓▓
```

### Chunk size guidelines (starting points)

| Use case | Chunk size | Overlap | Why |
|----------|-----------|---------|-----|
| RAG Q&A / FAQ | 500–800 | 50–100 | Precision |
| Documentation | 1000–1500 | 200 | Keep context |
| Code | 1500–2000 | 200 | Whole functions stay intact |
| Books / articles | 800–1200 | 150 | Balanced |

> Guidelines only — tune to your use case. **Units are characters** with the default splitter (use `from_tiktoken_encoder` if you want these numbers to mean tokens).

### Chunk size in practice (`chunk_size_comparison`)

Smaller size → more chunks; larger size → fewer, fatter chunks. On the same ~889-char sample (overlap = 20% of size):

| `chunk_size` | # chunks | Trade-off |
|--------------|----------|-----------|
| 200 | 6 | Precise retrieval, but little context per chunk |
| 500 | 3 | **Sweet spot** — balanced |
| 1000 | 1 | One blob: lots of context, imprecise retrieval |

```python
for size in [200, 500, 1000]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=size // 5)
    print(size, len(splitter.split_text(sample_text)))
```

> There's **no universal best size** — experiment per document type, query patterns, and LLM context window. Chunk size directly impacts retrieval quality.

### Why overlap matters (`overlap_importance`)

Overlap repeats the tail of one chunk at the head of the next, so a thought split across a boundary isn't lost.

```python
no_overlap   = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
with_overlap = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20)
```

```
no overlap:    [ ...the lazy dog. ][ The quick brown... ]   (clean cut, context orphaned)
with overlap:  [ ...the lazy dog. ][ the lazy dog. The quick... ]   (shared phrase carried over)
```

**Why it's "cheap insurance":** without overlap, related info can land in different chunks:
- *Chunk 1:* "The API key expires after 24 hours." (mentions the problem, no fix)
- *Chunk 2:* "You must refresh it using the token endpoint." (the fix, doesn't mention expiration)

A query like *"how do I handle API expiration?"* matches Chunk 1 (has "expiration") but **misses the solution** in Chunk 2. With overlap, Chunk 2 also carries the "expires after 24 hours" phrase → the retriever pulls the chunk that has **both** problem and solution.

> Overlap ensures important phrases aren't orphaned at boundaries — the difference between finding *an* answer and finding a *complete* answer.

### Markdown header splitter (`markdown_splitter`)

Splits on **Markdown headers** instead of raw length — great for structured docs (READMEs, wikis, documentation). Each chunk keeps its header path as **metadata**, so you know *where* the content came from.

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
    ("#", "h1"), ("##", "h2"), ("###", "h3"),
])
chunks = splitter.split_text(sample_markdown)   # -> list[Document]

for c in chunks:
    print(c.metadata)        # e.g. {"h1": "Introduction to Machine Learning", "h2": "..."}
    print(c.page_content)
```

- Output is `Document`s whose `metadata` captures the header hierarchy (h1/h2/h3) each chunk sits under.
- **Preserves context, not just content** — a chunk "knows" it belongs to, say, *Introduction → Types → Unsupervised Learning*.
- Use when documents have structure; combine with a length splitter afterward if sections are still too big.

### Code splitter (`code_splitter`)

Code needs **syntax-aware** splitting so functions/classes stay intact instead of being cut mid-body. Use `RecursiveCharacterTextSplitter.from_language(...)` with a `Language` enum.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,          # also JAVA, C, COBOL, JS, ...
    chunk_size=500, chunk_overlap=50,
)
chunks = splitter.split_text(sample_code)
```

- `from_language` picks separators tuned to that language (defs, classes, blocks) → each chunk keeps a **whole function/class** with its name.
- On the sample, two functions → two clean chunks, each retaining its definition → a code query retrieves a coherent, complete function.
- The key move: **always pass the language** so the splitter respects syntax and doesn't cut mid-function.

### End-to-end: loader → splitter (`document_splitter`)

The real-world flow — load real files into `Document`s, then split them with `split_documents` (metadata carries through automatically).

```python
from langchain_community.document_loaders import PyPDFLoader

docs = PyPDFLoader("./docs/langchain_demo.pdf").load()   # 3 pages -> 3 Documents

splitter   = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)              # -> 9 Document chunks

print(len(docs), "->", len(split_docs))       # 3 -> 9
print(split_docs[0].metadata)                 # source, page, total_pages, producer, ...
print(split_docs[0].page_content[:200])
```

- `split_documents` (not `split_text`) is what you use once data is loaded — it keeps each chunk's `metadata` (source, page, etc.) from the loader.
- Loader + splitter compose cleanly: `PyPDFLoader` gives per-page `Document`s, the splitter breaks those into the ~500-char chunks you'll embed.
- This is exactly the shape feeding the RAG indexing pipeline → [05-rag-and-embeddings.md](05-rag-and-embeddings.md).

### Semantic chunking

Instead of fixed sizes, **group sentences by meaning**: embed sentences, measure similarity, cluster related ones together.

```
mixed doc (ML + food + ...) ──▶ semantic chunker ──▶ [ML chunk] [food chunk] ...
```

- Best when a document mixes **distinct topics**.

**Why it costs embedding API calls:**
- Fixed-size splitters (`RecursiveCharacterTextSplitter`) are purely **mechanical** — they count characters and cut on separators. No model, no network → free and instant, runs fully local.
- Semantic chunking must **embed every sentence first** to compare their meanings and decide where topics change. If your embedding model is hosted (e.g. OpenAI), each of those embeddings is a **billed API call** — so you pay money + wait on network latency, at index time, across the whole corpus.
- That's *extra* embedding work: you embed sentences to find boundaries, then still embed the final chunks to store them in the vector DB.

**So:** only reach for semantic chunking when cleaner topic boundaries actually improve retrieval (mixed-topic docs). For most cases, fixed-size splitting is cheaper, faster, and good enough. (A **local** embedding model — see [05-rag-and-embeddings.md](05-rag-and-embeddings.md) — removes the per-call cost but not the added compute/time.)

*TODO: add the splitter code once the course implements it (this section is the concept overview).*
