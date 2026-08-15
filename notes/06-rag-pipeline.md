# 6. RAG Pipeline

> **Quick recall:** Wire the pieces together — a **retriever** fetches relevant docs, the **question** passes through unchanged, both fill a **grounding prompt**, and the **LLM** answers *only* from that context. Grounding + "say I don't know" reduces hallucination; returning **sources** lets users verify.

Builds on [05-rag-and-embeddings.md](05-rag-and-embeddings.md) (embeddings, vector store, retriever).

---

## Architecture recap

```
user query ─┬─────────────▶ retriever ─▶ relevant docs (context)
            │                                     │
            └─────────────────────────────────────┤
                                                   ▼
                                  prompt(context + question)
                                                   ▼
                                            LLM ─▶ grounded answer
```

- Driven by the **vector store**: docs are embedded + indexed for search.
- **Key point:** RAG grounds the LLM's answer in real documents → less hallucination.

## Basic RAG chain (LCEL)

The classic shape — context and question in parallel, then prompt → LLM → parser:

```python
chain = (
    {"context": retriever, "question": RunnablePassthrough()}   # parallel inputs
    | prompt
    | llm
    | StrOutputParser()
)
```

- **context** comes from the `retriever` (the *retrieval* + *augmentation* in RAG).
- **question** goes through `RunnablePassthrough()` — kept original/unchanged (see [03-working-with-llms.md](03-working-with-llms.md)).
- Both feed the prompt together — context and question go hand in hand.

### How the dict becomes a `RunnableParallel`

You never call `RunnableParallel` here — LCEL **coerces** the plain dict into one, because it sits inside a `|` pipe. When you write `a | b`, LangChain runs `a.__or__(b)` and passes any non-Runnable operand through an internal `coerce_to_runnable()`:

| What you write | LCEL turns it into |
|----------------|--------------------|
| a `dict` | `RunnableParallel` |
| a plain function (`format_docs`) | `RunnableLambda` |
| already a Runnable (`retriever`, `prompt`) | left as-is |

So `{"context": ..., "question": ...} | prompt` becomes `RunnableParallel({...}) | prompt` before anything runs.

**What `RunnableParallel` does at runtime:** every branch receives the **same single input** and they run **concurrently**; the output is a dict with the **same keys**, each mapped to its branch's result.

For `rag_chain.invoke("Who created LangChain?")`:

```
input = "Who created LangChain?"   (one string, sent to BOTH branches)

        ┌──────────────────────────────────────────────┐
"Who    │  "context":  retriever | format_docs          │
created │       → retriever → [Document, Document]       │  ← run in
Lang    │       → format_docs → "LangChain is a ..."     │    parallel
Chain?" │  "question":  RunnablePassthrough()            │
        │       → "Who created LangChain?" (unchanged)   │
        └──────────────────────────────────────────────┘
                          ▼
   {"context":  "LangChain is a framework... by Harrison Chase...",
    "question": "Who created LangChain?"}
                          ▼
                prompt  (fills {context} and {question})
```

- The branch **keys** (`context`, `question`) are deliberately the **same names** as the prompt's `{context}` / `{question}` — that's why the output dict slots straight in.
- `RunnableParallel` is the **fan-out** that turns the one input from `invoke` into the two-key dict the prompt needs: one branch transforms it (retrieve + format), the other forwards it untouched.

These two lines are identical — the dict form is just the idiomatic shorthand (which is why the file imports `RunnableParallel` but never writes it):

```python
{"context": retriever | format_docs, "question": RunnablePassthrough()}          # implicit
RunnableParallel(context=retriever | format_docs, question=RunnablePassthrough()) # explicit
```

### In code (`013_rag_pipeline.py`)

**1. Build the knowledge base** (`create_kb`) — split → embed → store, returns a vector store:

```python
def create_kb():
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc      = Document(page_content=KNOWLEDGE_BASE, metadata={"source": "langchain_knowledge_base.md"})
    chunks   = splitter.split_documents([doc])
    return Chroma.from_documents(documents=chunks, embedding=embeddings_model,
                                 persist_directory=tempfile.mkdtemp())
```

**2. The RAG chain** (`demo_basic_rag`):

```python
retriever = create_kb().as_retriever(search_type="similarity", search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_template(
    "Answer the question based only on the following context:\n\n{context}\n\n"
    "Question: {question}\n\nAnswer:\n"
    'Answer concisely, and if you don\'t know, just say "I don\'t know."')

def format_docs(docs):                       # join retrieved chunks into one string
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("Who created LangChain?")   # -> grounded answer string
```

- **`retriever | format_docs`** — the retriever returns `list[Document]`; `format_docs` (a plain function, auto-wrapped as a `RunnableLambda` in the pipe) flattens them into the `{context}` string.
- The input dict keys (`context`, `question`) match the prompt's `{variables}`.
- `invoke(question)` takes a **plain string** — `RunnablePassthrough` forwards it to `{question}`, and it's also what the retriever searches with.
- Flow: `question → retrieve+format context / passthrough question → prompt → llm → string`.

### What `format_docs` does

An adapter between the retriever and the prompt. The retriever outputs `list[Document]`, but the prompt's `{context}` hole needs **text** — so `format_docs` extracts each chunk's text and glues it into one string.

```python
def format_docs(docs):                       # docs = list[Document] from the retriever
    return "\n\n".join(                       # glue the pieces with a blank line between each
        doc.page_content for doc in docs      # pull just the TEXT out of each Document
    )
```

- A `Document` has `.page_content` (the chunk text) **and** `.metadata` (source, etc.); this keeps only `.page_content`.
- `"\n\n".join(...)` merges the chunks into **one** block, separated by a blank line so they stay readable.

```
retriever output                       format_docs            {context}
[Document(page_content="A"),  ──────────────────────▶  "A\n\nB\n\nC"  (one string)
 Document(page_content="B"),
 Document(page_content="C")]
```

> The sources variant (`format_docs_with_sources`) is the same idea but prefixes each chunk with its `[source]` from `.metadata` — see [RAG with sources](#rag-with-sources-citations).

## Grounding the prompt (anti-hallucination)

Instructions matter. A grounding pattern:

```
Answer based ONLY on the following context.
If the context doesn't contain the answer, say "I don't know."

Context: {context}
Question: {question}
```

- **Without instructions:** *"What is quantum computing?"* → the LLM confidently makes something up.
- **With instructions:** if the context lacks it → *"I don't know / I don't have information about that."*
- This is the low-hanging fruit of prompt engineering — cheap, easy, and prevents coherent-but-fabricated answers.

## RAG with sources (citations)

Return **where** each answer came from so users (and the system) can verify.

Same pipeline as basic RAG — the **only** change is that it keeps each chunk's `.metadata["source"]` (which basic RAG throws away) and puts it into the context so the LLM can cite it. Two small diffs vs `demo_basic_rag`:

| | `demo_basic_rag` | `demo_rag_with_sources` |
|---|---|---|
| **format function** | `format_docs` → only `.page_content` | `format_docs_with_sources` → tags each chunk with `[n] source:` |
| **prompt** | "answer concisely" | "answer... **include which sources you used**" |

The work happens in the format helper — it reads the metadata the basic version discards:

```python
def format_docs_with_sources(docs):
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")   # ← the source (basic RAG dropped this)
        formatted.append(f"[{i+1}] {source}:\n{doc.page_content}")
    return "\n\n".join(formatted)
```

```
retriever output:   page_content + metadata {"source": "kb.md"}
                          │  format_docs_with_sources()  (tags each chunk with its source)
                          ▼
formatted context:  [1] kb.md: <chunk text>
                    [2] guide.md: <chunk text>
```

The `{context}` the LLM now sees is labeled (`[1] kb.md: ...`) instead of bare text, giving it the labels to cite back. The chain wiring (retriever + passthrough → prompt → llm → parser) is otherwise identical.

- **Why sources matter:** users can verify answers (citations) → builds trust; the system also "knows" which document grounded its response.

## Structured RAG output (`demo_structured_rag`)

Instead of a free-text answer, force the model to return a **typed object** with fixed fields — useful when downstream code needs `answer`, `confidence`, `sources`, etc. as real values, not text to re-parse.

> **Pydantic in one look** — a **data-validation** library: you declare the *shape* of data as a class (types + rules) and it **enforces** that shape at runtime. It's "type hints that actually do something" — plain Python ignores `age: int`; Pydantic checks it, converts when safe (`"30"→30`), or raises a clear `ValidationError`.
> - **`BaseModel`** = subclass it to declare one data shape; **`Field`** = attach rules/metadata to one attribute.
> - **Why LangChain uses it:** it can export the class as a **JSON Schema** (`model_json_schema()`). That does double duty in `with_structured_output`: (1) *before* the call the schema tells the LLM exactly what fields/types/descriptions to produce; (2) *after*, it validates the model's JSON into a real object — guaranteed fields, no manual parsing.
> - **Mental model:** the **contract enforcer** at your program's boundary — past a `BaseModel`, messy external data (API, config, **LLM output**) is exactly the shape your code expects.
>
> Full deep dive (field types, validators, load/dump, JSON Schema) → [pydantic-reference.md](pydantic-reference.md).

Define the shape with a **Pydantic `BaseModel`**:

```python
from pydantic import BaseModel, Field

class RAGResponse(BaseModel):
    """Structured RAG response."""
    answer: str        = Field(description="The answer to the question")
    confidence: str    = Field(description="high, medium, or low")
    sources_used: List[str] = Field(description="List of sources referenced")
    follow_up: str     = Field(description="Suggested follow-up question")
```

**What `BaseModel` is:** the base class (from **Pydantic**) you subclass to declare a **typed, self-validating schema** — each attribute has a type, and Pydantic checks incoming data against it. Here it defines the exact shape you want back.

- Each field = `name: type = Field(description="...")`. The **type** (`str`, `List[str]`) enforces structure; the **`description`** is fed to the LLM so it knows what to fill in.
- The docstring + descriptions become part of the schema the model sees — treat them as instructions, not comments.

**What `Field` is:** a Pydantic helper that attaches **metadata/rules** to a single attribute — beyond its bare type. Here it's used for the `description`, which becomes the LLM's instruction for *that* slot (e.g. `confidence` should be `"high, medium, or low"`). Better descriptions → better-filled fields.

```python
answer: str = Field(description="The answer to the question")
#      │              │
#      type           metadata for this field (read by the LLM)
```

| `Field` can also set | Example |
|----------------------|---------|
| default value | `Field(default="unknown")` |
| validation constraints | `Field(ge=0, le=1)`, `Field(min_length=1)` |
| JSON alias | `Field(alias="user_id")` |

> Rule of thumb for structured output: **always give each field a `description`** — the model treats it like a mini-prompt for that slot. (No default → the field is required.)

Wire it in with `with_structured_output` (note the chain ends at the model — **no `StrOutputParser`**, since the output is an object, not a string):

```python
structured_llm = llm.with_structured_output(RAGResponse)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | structured_llm                     # ← returns a RAGResponse, not text
)

result = rag_chain.invoke("What is LangGraph?")
result.answer        # -> str        (typed attributes, no parsing)
result.confidence    # -> "high"
result.sources_used  # -> ["langchain_knowledge_base.md"]
```

- `with_structured_output(RAGResponse)` makes the model emit a **validated `RAGResponse` instance** — guaranteed fields, correct types, no string-wrangling.
- Same as `with_structured_output` in [03-working-with-llms.md](03-working-with-llms.md); the agent's `response_format` in [07-agents.md](07-agents.md) does the equivalent for agents.

### Who fills in the values? — the LLM does

You define the *form* (fields, types, descriptions); the **LLM fills it in**, using the retrieved context to decide each value. Under the hood `with_structured_output` uses the model's **tool/function-calling**: it converts the class to a JSON Schema and instructs the model *"don't reply in free text — return values matching exactly these fields."*

```
RAGResponse ──model_json_schema()──▶ { answer: str, confidence: str,
                                       sources_used: str[], follow_up: str }
        │  handed to the model as a "tool" it must fill
        ▼
LLM reads context + question, EMITS a value per field:
        { "answer": "LangGraph builds stateful, multi-actor apps.",
          "confidence": "high",
          "sources_used": ["langchain_knowledge_base.md"],
          "follow_up": "How does LangGraph handle persistence?" }
        ▼
Pydantic VALIDATES the JSON ──▶ RAGResponse(answer=..., confidence=..., ...)
```

| Thing | Decided by |
|-------|-----------|
| which fields exist + their types | **you** (the Pydantic class) |
| the guidance for each field | **you** (`Field(description=...)`) |
| the actual value in each field | **the LLM** (reads context + descriptions, generates it) |
| whether those values are valid | **Pydantic** (validates before you get the object) |

> The `Field(description=...)` is the steering wheel — it's a **mini-prompt per slot**. The model sees `"high, medium, or low"` and picks one for `confidence`. Vague description → vaguely-filled field.

---

*Next: build the RAG pipeline from scratch in code.*
