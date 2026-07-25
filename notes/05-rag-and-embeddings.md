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

---

*Next: storing vectors and building the RAG pipeline (vector store, retrieval).*
