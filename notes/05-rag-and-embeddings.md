# 4. RAG & Vector Embeddings

> **Quick recall:** A raw LLM only knows its training data (fixed cutoff, no private data, can hallucinate). **RAG** = retrieve relevant chunks of *your* documents and feed them to the LLM as context before it answers. Three phases: **Index** (once) → **Retrieve** (per query) → **Generate**.

---

## Why RAG?

Raw LLM limitations:
- **Knowledge cutoff** — trained up to a date, unaware of anything newer.
- **No private data** — never seen your docs, wikis, DBs.
- **Hallucination** — with no context, it may confidently make things up.

```
Plain LLM:   question ─────────────▶ LLM ─▶ answer   (limited, ungrounded)

RAG:         question ─▶ retrieve your docs ─▶ LLM(question + context) ─▶ grounded answer
```

**Core idea:** give the LLM the information it needs *before* it answers.

## The three phases

```
① INDEXING  (one-time setup)
   documents ─▶ chunk ─▶ embedding model ─▶ vectors ─▶ vector DB

② RETRIEVAL (per query)
   question ─▶ embedding model ─▶ vector ─▶ similarity search ─▶ relevant chunks

③ GENERATION
   question + relevant chunks ─▶ LLM ─▶ grounded answer
```

Retrieval + feeding context into the prompt = the **augmentation** in *Retrieval-Augmented Generation*.

## Indexing (the quality phase)

`gather → clean → extract → chunk → embed → store`

- Handles **unstructured / raw data**: PDF, TXT, MD, CSV, ...
- **Garbage in, garbage out** — clean, current, well-organized input is critical; this is one of the most important phases.

**Library analogy:** gather books → discard damaged/contradictory ones → keep and organize the best → librarian can find them fast. Clean, current, well-organized shelves = good retrieval.

## Chunking

Break documents into smaller, **semantically coherent** pieces (e.g. a 100-page manual → per-section chunks, each on one topic).

**Why it matters:**
- **Embedding models have token limits** — can't ingest a whole book at once.
- **LLMs have a finite context window** — can't pass unlimited tokens.
- Small, topic-focused chunks → retrieve *only* what's relevant → coherent answers.

Each good chunk should:
- fit within the embedding model's token limit,
- contain **one complete thought/concept**,
- stand alone (enough context to be understood independently).

> Bad chunking → broken RAG. Good chunking → good RAG. Chunk size must be *just right*.

## RAG vs alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **Fine-tuning** | Deep customization (data baked into model internals) | Expensive, slow, needs ML expertise |
| **Prompt engineering** | Free, fast, anyone can do it | Limited by context window |
| **RAG** | Best of both; grounded & controllable | Needs a vector DB setup (simple) |

## RAG advantages

- **Up-to-date** — feed new data anytime, no retraining.
- **Transparent** — you can see which documents were used.
- **Cost-effective** — no model retraining.
- **Accurate** — grounded in real sources (your documents).
- **Scalable** — millions of documents, still retrieves the relevant few.

---

## Vectors

A vector = **direction + magnitude** (e.g. "toward Camp 2, 3.4 miles"). In practice it's just an **array of numbers** — like GPS coordinates pinpointing a location in space.

```
2D: [40.7, -74.0]              ← position on a map (2 values)
3D: [x, y, z]                  ← 3 values
AI: [0.12, -0.88, ..., 0.31]   ← 100s–1000s of dims (e.g. 1536)
```

- **Each dimension captures a different aspect of meaning.** More dimensions → more meaning/features represented.
- Analogy — a house as a 7-D vector `[3, 2, 1500, 2005, ...]` = bedrooms, bathrooms, sq ft, year built...
- **Vectors close together = similar meaning** (semantic similarity). e.g. "lake" and "water" sit near each other.

## Similarity = distance

Measure how close two vectors are:

| Metric | Meaning | Scale |
|--------|---------|-------|
| **Cosine similarity** | angle between vectors | −1 → 1 (`1` identical · `0` unrelated · `−1` opposite) |
| **Euclidean distance** | straight-line distance | smaller = more similar |

```
dog   [0.9, 0.8, 0.1]
puppy [0.9, 0.7, 0.2]   cosine(dog, puppy) ≈ 0.92  → very similar
car   [0.1, 0.2, 0.9]   cosine(dog, car)   ≈ 0.23  → unrelated
```

This is how AI "knows" dog ~ puppy but dog ≠ car — purely from the numbers.

## Embeddings

**Embeddings = vectors (+ a little metadata) that capture semantic meaning.** An embedding model converts text → vector.

```
text chunks ──▶ embedding model ──▶ semantic vectors ──▶ vector DB
```

- **Word embeddings** — "king" & "queen" land closer to each other than to "cat".
- **Sentence embeddings** — whole sentences too; pizza sentences cluster away from random ones.
- What's stored is **semantic meaning**, not raw text — exactly what similarity search needs downstream.

---

## Embeddings in code

Instantiate an embedding model, then turn text into vectors (`010_embeddings.py`):

```python
from langchain_openai.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# One string → one vector
vec = embeddings.embed_query("This is a test document.")
print(len(vec))          # 1536  (the model's dimension)

# Many strings → list of vectors (batch)
vecs = embeddings.embed_documents([
    "This is the first test document.",
    "This is the second test document.",
])
print(len(vecs), len(vecs[0]))   # 2 vectors, each 1536 dims
```

**Two methods:**

| Method | Input | Output |
|--------|-------|--------|
| `embed_query(text)` | one string (the user's question) | one vector |
| `embed_documents([...])` | list of strings (your chunks) | list of vectors |

### Choosing an OpenAI embedding model

| Model | Dimensions | Notes |
|-------|-----------|-------|
| `text-embedding-3-small` | 1536 | Cheap, general-purpose default |
| `text-embedding-3-large` | 3072 | Higher accuracy, more cost |
| `text-embedding-ada-002` | 1536 | Deprecated |

- More dimensions → more captured meaning, but higher cost.
- `dimensions=` param can override the default size for some models.
- The raw output looks like gibberish floats — it only "makes sense" once stored in a vector DB and compared via similarity search.

### Local / free embedding models

OpenAI charges per token. To run **local, zero-cost** embeddings, swap the wrapper — the `embed_query` / `embed_documents` interface is identical (that's the payoff of LangChain wrappers).

```python
# HuggingFace sentence-transformers (runs locally)
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # 384 dims

# Ollama (local server)   →  uv add langchain-ollama
from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

- `all-MiniLM-L6-v2` = only **384 dims** → less captured meaning, but fine for small/test workloads.
- Tradeoff stays the same: fewer dims = cheaper/faster but coarser; ~1536 is the common standard.
- Code downstream is unchanged — only the model instantiation line differs.

## Embeddings deep dive (`011_embeddings_deep.py`)

### Single embedding + normalization

```python
import numpy as np
vec = embedder.embed_query("What is Machine Learning?")

len(vec)                      # 1536
np.linalg.norm(vec)           # 1.0  -> the vector is normalized
```

- **Vector norm = magnitude** (length): `norm(v) = √(v₁² + v₂² + ... )`.
- **Normalizing** = scale the vector to length 1 by dividing each component by its norm — same direction, length becomes 1 (a "unit vector").
- Norm ≈ 1 → normalized; ≠ 1 → not (some models don't normalize).

**Worked example** — two texts, same meaning, one wordier:

```
Doc A "dog"             → [3, 4]   norm = √(9+16)  = 5
Doc B "dog dog dog dog" → [6, 8]   norm = √(36+64) = 10   (same direction, bigger only due to length)

normalize (÷ own norm):
A: [3,4] ÷ 5  → [0.6, 0.8]
B: [6,8] ÷ 10 → [0.6, 0.8]   ← identical!
```

Normalizing strips out *how much text* so meaning is compared fairly — you can't look "more similar" just by writing more. Bonus: with unit vectors, **cosine similarity = plain dot product** (the `norm·norm` denominator is `1·1`).

### Batch embedding

Same interface as one call, just pass a list — `embed_documents` takes `list[str]`:

```python
vecs = embedder.embed_documents([
    "What is Machine Learning?",
    "Explain overfitting in ML.",
    "How does a neural network work?",
])   # -> list of vectors, one per input
```

### Similarity search (mini "pre-RAG")

Embed docs + query, score each doc against the query with **cosine similarity**, rank:

```python
docs  = ["Python is a programming language",
         "JavaScript is used for web development",
         "Machine learning enables AI applications",
         "Cats are popular pets"]
query = "What programming languages exist?"

doc_vecs   = embedder.embed_documents(docs)
query_vec  = embedder.embed_query(query)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

scores = [cosine_similarity(query_vec, d) for d in doc_vecs]
ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
```

- The programming-language docs rank top; "Cats are popular pets" sinks to the bottom — retrieval by **meaning**, not keywords.
- This is exactly what a vector store's retriever does under the hood (just at scale, with an index). It's the **retrieval** half of RAG in miniature.

### Embedding caching (`embedding_caching`)

Embedding text costs API calls — **cache** results so re-embedding the same text is free.

```python
from langchain_classic.embeddings.cache import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore

store = LocalFileStore(root_path=tempdir)          # persists cache to disk
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embedder,                # the real model
    document_embedding_cache=store,
    namespace="exercise",                          # separates different caches
)

vectors1 = cached_embedder.embed_documents([text])   # 1st call → hits the API, stores result
vectors2 = cached_embedder.embed_documents([text])   # 2nd call → served from cache
np.allclose(vectors1[0], vectors2[0])                # True — identical vectors
```

- Wraps a real embeddings model + a store; transparent — same `embed_documents` interface.
- `namespace` keeps caches from different models/runs separate.
- ⚠️ `CacheBackedEmbeddings` moved to **`langchain-classic`** in LangChain 1.0 (deprecated ~Dec 2026); the concept stays, only the import path will change.

---

## Vector stores

The **indexing → retrieval** pipeline, now with persistence:

```
INDEX:   chunks ─▶ embeddings ─▶ vector store (persisted)
RETRIEVE: query ─▶ embed ─▶ similarity_search(k) ─▶ top-k relevant chunks
```

**Chroma** — the local-dev default: free, no setup (`uv add langchain-chroma`), persists to disk. Great for prototyping/small data (but also scales beyond that).

### Key vector-store methods

| Method | Purpose |
|--------|---------|
| `add_documents(docs)` | **Write** new documents (embeds + stores them) |
| `similarity_search(query, k=4)` | **Read** — return the `k` most similar chunks |
| `similarity_search_with_score(query, k)` | Same, but returns `(doc, score)` tuples |
| `as_retriever(search_kwargs=...)` | Expose the store as a **retriever** to plug into LCEL/RAG chains |

- `k` = how many similar docs to return (tune it).
- `as_retriever()` is the bridge from vector store → RAG chain (used with `RunnablePassthrough`, see [03-working-with-llms.md](03-working-with-llms.md)).

### Production options

| Store | Best for | Cost | Type |
|-------|----------|------|------|
| **Chroma** | Local dev / prototyping | Free | Local |
| **FAISS** (Meta) | Speed, in-memory play | Free | Local (in-memory) |
| **Pinecone** | Production, managed | ~$70+/mo (free tier exists) | Cloud |
| **Weaviate** | Self-hosted | Free / paid | Hybrid |
| **Qdrant** | Performance + strong filtering | Free / paid | Hybrid |

> Start with **Chroma** locally; move to **Pinecone** or **Qdrant** for production scale.

## Chroma in code (`012_vector_stores.py`)

Install: `uv add langchain-chroma`.

> ⚠️ **Python 3.14 + Chroma:** Chroma depends on Pydantic v1, which isn't yet compatible with 3.14's typing internals. Use **Python 3.11 or 3.12** for the Chroma sections (`uv venv --python 3.12`). Don't debug it for hours — just switch the interpreter version and rerun.

### Create a store + search (`chroma_basics`)

```python
from langchain_chroma import Chroma

# build store from documents — embeds + persists in one call
vectorstore = Chroma.from_documents(
    documents=SAMPLE_DOCS,          # list[Document]
    embedding=embedder,             # embeds each doc automatically
    persist_directory=tmpdir,       # saved to disk
)
print(vectorstore._collection.count())     # how many docs are stored

# retrieve the k most similar
results = vectorstore.similarity_search("What is LangChain?", k=2)
for doc in results:
    print(doc.page_content, doc.metadata["source"])
```

- **`from_documents(documents, embedding, persist_directory)`** — the one-shot builder: embeds every doc and writes the store to disk. (There's also `from_texts` for raw strings.)
- Embedding happens **automatically** — you pass the model, Chroma calls it.
- `similarity_search(query, k)` returns the top-`k` relevant `Document`s (metadata preserved).
- No need to hand-write cosine similarity (unlike the [pre-RAG demo](#embeddings-deep-dive-011_embeddings_deeppy)) — the store's method does it.

### Similarity search with scores (`similarity_search_with_scores`)

Returns `(document, score)` tuples so you can see *how* relevant each hit is.

```python
results = vectorstore.similarity_search_with_score("Explain vector stores.", k=3)
for doc, score in results:
    similarity = 1 / (1 + score)          # Chroma returns a distance → convert to 0–1 similarity
    print(similarity, doc.page_content)
```

- Chroma's `score` is a **distance** (lower = closer). `1 / (1 + distance)` maps it to a 0–1 similarity for readability.

### Metadata filtering (`metadata_filtering`)

Restrict the search to docs whose metadata matches — combine semantic search *and* structured filters.

```python
# without a filter: k=5 across everything
vectorstore.similarity_search("What databases are available?", k=5)

# with a filter: only docs whose metadata topic == "database"
vectorstore.similarity_search(
    "What databases are available?", k=5,
    filter={"topic": "database"},
)
```

- `filter={...}` is checked against each doc's `metadata`; only matches are eligible.
- Narrows results to the relevant subset — e.g. drops a `topic: "architecture"` doc even if it's semantically close.
- Powerful for scoping retrieval (by source, topic, date, etc.) on top of similarity.

### Persistence — save & reload (`persist_chroma`)

Persist to a real directory, then reload from disk later (survives restarts) without re-embedding.

```python
persist_dir = "./chroma_db/"

# create + persist
vectorstore = Chroma.from_documents(
    documents=SAMPLE_DOCS, embedding=embedder, persist_directory=persist_dir)

del vectorstore                       # simulate a restart

# reload from disk — note the DIFFERENT kwarg name
reloaded = Chroma(
    embedding_function=embedder,      # ← constructor uses `embedding_function`
    persist_directory=persist_dir,
)
reloaded.similarity_search("LangChain", k=2)   # works, no re-embedding
```

- ⚠️ **Kwarg gotcha:** `Chroma.from_documents(...)` takes `embedding=`, but the `Chroma(...)` **constructor** (used to reload) takes `embedding_function=`. Mixing them up is a common error.
- Reloaded store keeps the same doc count — embeddings were saved, not recomputed.
- **Under the hood:** Chroma persists as a **SQLite** DB in `persist_dir`. You can inspect it (e.g. a SQLite viewer) — the collection is named `langchain`, with the embedding `dimensions` (1536 for `text-embedding-3-small`), plus tables for documents, metadata, and embeddings.

### How Chroma stores data (SQLite + HNSW)

```
chroma_db/
├── chroma.sqlite3          # relational: collections, doc IDs, text, metadata, doc↔vector map
└── <collection_uuid>/      # one folder per collection = the HNSW vector index
    ├── header.bin          # HNSW config & index metadata
    ├── data_level0.bin     # the actual embedding vectors
    ├── link_lists.bin      # graph edges (each vector's neighbors)
    └── length.bin          # offsets/lengths to access vectors & graph fast
```

- **SQLite** = the "metadata database" (documents, metadata, IDs, doc→vector mapping).
- **`<uuid>/*.bin`** = the vector index (vectors + the graph that connects them).

**ANN (Approximate Nearest Neighbor)** — instead of comparing the query against *every* vector (brute force over millions), navigate an index and compare only a handful of candidates → nearly identical results, far faster. "Approximate" = very close to the true nearest, much quicker.

**HNSW (Hierarchical Navigable Small World)** — the graph-based ANN algorithm Chroma uses. Each vector is a **node** storing its embedding + links to similar neighbors. Search:

```
query ─▶ start at an entry node ─▶ hop to a more-similar neighbor ─▶ repeat
      until no neighbor is closer ─▶ return top-K
```

**Takeaways:** SQLite stores docs/metadata/IDs · `.bin` files store vectors + graph links · ANN trades a tiny bit of accuracy for big speed · HNSW is the popular multi-layer-graph ANN that powers it.

### As a retriever (`as_retriever`)

`as_retriever()` turns the store into a **runnable** — it has `.invoke()`, so it drops straight into LCEL/RAG chains (this is the bridge to note 03's pass-through RAG shape).

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)
docs = retriever.invoke("How do I build AI applications?")   # -> list[Document]
```

- Same relevance results as `similarity_search`, but as a composable runnable you pipe into a chain.

**MMR (Maximal Marginal Relevance)** — returns **diverse** results instead of near-duplicates:

```python
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 5},   # fetch 5 candidates, return 3 diverse ones
)
```

- `fetch_k` = how many candidates to pull before picking the `k` most **diverse** among them.
- **similarity vs MMR:** plain similarity can be an *echo chamber* (three near-identical hits); MMR gives **breadth** — e.g. the concept + the foundation + a practical tool.
- **When to use:** similarity → you just want the top-most relevant; MMR → you want a well-rounded answer covering different angles.

### Setup exercise (`exercise_vector_store_setup`)

End-to-end helper: raw strings → chunks → Chroma → configured retriever.

```python
def create_retriever(texts, chunk_size=500, chunk_overlap=50, k=3):
    docs = [Document(page_content=t) for t in texts]
    split_docs = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_documents(docs)
    vectorstore = Chroma.from_documents(documents=split_docs, embedding=embedder)   # in-memory (no persist_directory)
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
```

- Ties the whole pipeline together: **load → split → embed/store → retrieve**.
- Omitting `persist_directory` keeps the store **in-memory** (fine for quick tests).

---

*This completes the vector store fundamentals → next: wiring a retriever into a full RAG chain (retrieve → prompt → LLM).*
