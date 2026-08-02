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
