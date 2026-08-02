# Agents with LangChain & LangGraph — Notes

Quick-refresh reference docs. Read top-to-bottom to reload full context.

| # | Note | Covers |
|---|------|--------|
| 1 | [01-setup.md](01-setup.md) | Project setup: uv, env, API keys, first LLM call |
| 2 | [02-basic-langchain.md](02-basic-langchain.md) | V1 architecture, runnables, messages, prompt templates, few-shot, structured output |
| 3 | [03-working-with-llms.md](03-working-with-llms.md) | Model config, message objects & types, multi-turn, prompt templates, MessagesPlaceholder, few-shot, composition; LCEL chains, parallel/passthrough/branch, debugging |
| 4 | [04-document-loading.md](04-document-loading.md) | Document loaders (PDF/text/web/directory), Document objects, PDF loader options, chunking & text splitters |
| 5 | [05-rag-and-embeddings.md](05-rag-and-embeddings.md) | RAG overview: why RAG, index/retrieve/generate, chunking, vs alternatives; vectors, similarity, embeddings + code; Chroma vector store |
| 6 | [06-rag-pipeline.md](06-rag-pipeline.md) | Assembling the RAG chain: retriever + passthrough → grounding prompt → LLM; anti-hallucination, sources/citations |
| — | [imports-reference.md](imports-reference.md) | Every import used across the `.py` files and why — kept in sync with the code |
