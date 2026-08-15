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

> **Go-to strategy for code:** `RecursiveCharacterTextSplitter.from_language(language=Language.X, ...)` — a **language-aware** recursive splitter that keeps functions/classes/logical blocks whole. Pair with a **larger chunk size (~1500–2000)** so full functions fit. (LangChain also has `Language`-specific splitters, but `from_language` on the recursive splitter is the standard default.)

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

**Overlap in code** — plays a smaller role than in prose, because `from_language` already splits at syntactic boundaries (between functions/classes), so it rarely cuts mid-function on its own. It still earns its place when:
- **A function is larger than `chunk_size`** → the splitter *must* cut mid-body; overlap keeps continuity (a variable/loop opened in chunk 1 is still visible at the start of chunk 2).
- **Carrying shared context** → imports, a class header, or a helper's signature bleed into the next chunk so a retrieved chunk isn't orphaned from what it depends on.

> ⚠️ Code is **token-dense** — too much overlap duplicates code across chunks (wasted tokens, near-identical hits). Keep it modest (~200): boundaries do most of the work; overlap is the safety net for functions that don't fit in one chunk.

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

**How it works** (unlike fixed-size, the *content* decides the cuts):

```
1. Split the doc into sentences
2. Embed each sentence (→ a vector per sentence)
3. Walk sentence-to-sentence, measuring similarity (cosine distance) between neighbours
4. Where distance SPIKES = topics shifted = a breakpoint → start a new chunk
5. Consecutive similar sentences get grouped into one chunk
```

```
sentence:    s1   s2   s3   │   s4   s5   │   s6
distance:      .1   .1   .8 ↑   .1   .7 ↑   .1
                        break      break
result:     [ s1 s2 s3 ]   [ s4 s5 ]   [ s6 ... ]
```

- A small **buffer/window** (combine a sentence with its neighbours before embedding) smooths out noise so a single odd sentence doesn't trigger a false break.
- **Breakpoint threshold** decides how big a distance jump counts as a split — LangChain's `SemanticChunker` supports `percentile` (default), `standard_deviation`, `interquartile`, and `gradient`.

**Implementations:**
- `SemanticChunker` from `langchain_experimental.text_splitter` — pass it an **embeddings model**; splits on the breakpoint threshold above.
- Provider splitters like `AI21SemanticTextSplitter` (managed, no local embedding step).

**Trade-offs to know:**
- **Variable chunk sizes** — chunks are as big as a topic runs; you may still cap/merge to a max size.
- Best for **heterogeneous** docs; for uniform single-topic text it's overkill.

**When to use it:**

| ✅ Use semantic chunking | ❌ Skip it (use fixed-size / structural) |
|--------------------------|------------------------------------------|
| Doc **mixes several topics** with no clear markers (transcripts, meeting notes, wandering articles, mixed knowledge bases) | Doc is **uniform / single-topic** → fixed-size is fine |
| Topic shifts **don't line up** with sizes or separators | Doc has **clear structure** → Markdown headers → `MarkdownHeaderTextSplitter`; code → `from_language` |
| **Retrieval precision** matters more than indexing cost | **Cost / speed / large scale** matters (the extra embedding pass hurts) |
| Fixed-size chunks were pulling in **mixed/irrelevant** content | You need **predictable** chunk sizes |

> Rule of thumb: reach for semantic chunking only when structure-based splitters (headers, `from_language`) don't apply **and** the document genuinely jumps between topics. Otherwise the cheaper splitters win.

**Why it costs embedding API calls:**
- Fixed-size splitters (`RecursiveCharacterTextSplitter`) are purely **mechanical** — they count characters and cut on separators. No model, no network → free and instant, runs fully local.
- Semantic chunking must **embed every sentence first** to compare their meanings and decide where topics change. If your embedding model is hosted (e.g. OpenAI), each of those embeddings is a **billed API call** — so you pay money + wait on network latency, at index time, across the whole corpus.
- That's *extra* embedding work: you embed sentences to find boundaries, then still embed the final chunks to store them in the vector DB.

**So:** only reach for semantic chunking when cleaner topic boundaries actually improve retrieval (mixed-topic docs). For most cases, fixed-size splitting is cheaper, faster, and good enough. (A **local** embedding model — see [05-rag-and-embeddings.md](05-rag-and-embeddings.md) — removes the per-call cost but not the added compute/time.)

### Chunking vs embedding — two separate steps

A common confusion: **splitting** and **embedding** are *different* steps, and **every** RAG pipeline embeds its chunks regardless of how they were split.

```
① SPLIT (cut text)                      ② EMBED (→ vector, stored)
  chunk = "def quicksort(arr): ..."  ──▶  [0.21, -0.44, ...]  ──▶ vector DB
```

- **Splitting** just decides *where the text is cut* (fixed-size, `from_language`, semantic…).
- **Embedding** is a **separate step that always happens**: each final chunk is passed to the embedding model → one vector → stored. This cost applies to **all** strategies — it's the base cost of RAG indexing. So yes, even fixed-size splitting costs you the embedding of the chunks.

**How a code chunk gets embedded:** once `from_language` gives you a chunk = one whole function, that **entire function's text** is fed to the embedding model, which returns **one vector** capturing its meaning (what it does, its identifiers, structure). A query like *"how do I sort a list?"* embeds to a nearby vector → the quicksort chunk is retrieved. The model reads code *as text* and encodes its semantics into that single vector.

**Does the model "understand" the code's context?** Partly — *semantically*, not *literally*:
- The embedding model is a **transformer**: **self-attention** means each token's representation is shaped by the tokens around it (that's the "context"), and it's pooled into one fixed-length vector. Trained on huge corpora that **include code**, it has learned what code is *about* (sorting, HTTP, recursion) from names, keywords, and idioms.
- ✅ So it captures the **gist** — but ❌ it does **not execute or reason** about the code; it recognizes `quicksort` relates to sorting because it has *seen* similar code, not because it traced the algorithm.
- What shifts the vector most: **identifiers, docstrings, comments** (good names → stronger signal; obfuscated `a`/`b`/`x` → weak). It only sees **what's in the chunk** — a helper defined elsewhere isn't "known" unless its text is present (why clean chunking + overlap matter). **Code-specialized** embedding models capture code semantics better than general ones.

**Example** — two chunks + a query (toy 3-D vectors; real ones ~1536-D):

```python
# Chunk A                          # Chunk B
def quicksort(arr):                def send_email(to, subject, body):
    """Sort a list ..."""              """Send an email via SMTP."""
```

```
query "how do I sort a list?"  →  [0.90, 0.10, 0.10]
Chunk A (quicksort)            →  [0.85, 0.20, 0.10]   cosine ≈ 0.98  ✅ retrieved
Chunk B (send_email)           →  [0.10, 0.20, 0.90]   cosine ≈ 0.30  ❌ not
```

The query lands near quicksort and far from send_email — purely from meaning; the model never ran the code, it recognized *sort/quicksort/arr* as "about sorting" from training.

Same algorithm, **obfuscated** (no names/docstring):

```python
def f(a):
    b = a[len(a) // 2]
    ...
```

```
Chunk A2 (obfuscated)  →  [0.55, 0.40, 0.30]   cosine ≈ 0.70  ⚠️ weaker match
```

Structure still hints at it, but the strong signal (`quicksort`, "Sort a list") is gone → shakier retrieval. **Good names + docstrings literally make code more findable.**

**The real difference in cost:**

| Strategy | Embedding rounds |
|----------|------------------|
| Fixed-size / `from_language` | **once** — only the final chunks (base cost) |
| Semantic | **twice** — sentences first (to find boundaries) **+** final chunks |

> So semantic chunking isn't "the one that costs embeddings" — *everything* embeds the final chunks. Semantic just adds an **extra** embedding pass up front.
>
> Tip: general text embedding models handle code reasonably (they saw code in training); **code-specialized** embedding models capture code semantics better if retrieval quality matters.
