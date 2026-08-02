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

```
retriever output:   page_content + metadata {"source": "doc.pdf"}
                          │  format_docs_with_source()  (tags each chunk with its source)
                          ▼
formatted context:  [Source: doc.pdf] <chunk text>
                    [Source: guide.md] <chunk text>
```

- A `format_docs` helper stitches retrieved chunks into the context string, tagging each with its `source`.
- **Why sources matter:** users can verify answers (citations) → builds trust; the system also "knows" which document grounded its response.

---

*Next: build the RAG pipeline from scratch in code.*
