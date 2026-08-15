# 2. Basic LangChain

> **Quick recall:** Everything in LangChain V1 is a **Runnable** with `invoke / batch / stream`. Compose them with **LCEL** (LangChain Expression Language) — the `|` pipe. Talk to chat models via **role-based messages**, not plain strings.

**Section covers:** LLMs & chat models · prompt templates · few-shot prompting · structured responses

---

## Why LangChain?

A raw LLM is just *text in → text out*. Real apps need much more around that call, and writing it by hand for every provider is repetitive and brittle. LangChain gives you **reusable building blocks with one common interface** so you can:

- **Swap providers** freely (OpenAI ↔ Anthropic ↔ local) without rewriting logic.
- **Structure prompts** (templates, few-shot, roles) instead of string-mashing.
- **Chain steps** together (prompt → model → parser → …) cleanly with LCEL.
- **Connect your data** via RAG (loaders, splitters, embeddings, vector stores).
- **Add memory, tools, streaming, retries, and tracing** without boilerplate.

**LangGraph** goes further: when a workflow needs **state, branching, loops/cycles, human-in-the-loop, or multiple agents**, LangGraph orchestrates it (LCEL chains are linear; LangGraph handles the complex, stateful graphs).

### What can you build?

| With LangChain (chains) | With LangGraph (stateful/agentic) |
|-------------------------|-----------------------------------|
| Chatbots & assistants | Tool-using agents (search, APIs, code) |
| RAG Q&A over your docs (PDF/wiki/DB) | Multi-agent systems & workflows |
| Summarize / extract / classify pipelines | Long-running flows with memory & retries |
| Semantic search | Human-in-the-loop approval steps |
| Content generation | Branching decision logic with cycles |

> Rule of thumb: **linear pipeline → LangChain (LCEL); complex, stateful, looping/agent workflow → LangGraph.**

---

## Which version am I on?

**"V1" = LangChain 1.x** — the modern framework, and what this project runs:

```
langchain 1.3.x · langchain-core 1.5.x · langchain-classic 1.0.x · langgraph 1.2.x
```

**Evolution in three phases:**

| Phase | What it looked like |
|-------|---------------------|
| **0.x (early)** | One monolithic `langchain` package; pipelines built with legacy classes (`LLMChain`, `SequentialChain`, `ConversationChain`). |
| **Modularization** | Split into focused packages: `langchain-core` (Runnables + LCEL), `langchain-openai`/`-anthropic` (providers), `langchain-community` (integrations). |
| **1.0 "V1" (now)** | LCEL + Runnables are **the** way to build; legacy chain classes moved to **`langchain-classic`** (backward-compat only). |

> **Takeaway:** Build everything with LCEL + Runnables (the `|` pipe). If a tutorial uses `LLMChain` / `SequentialChain`, that's pre-1.0 legacy — skip it. (`langchain-classic` still exists only for old code, e.g. `CacheBackedEmbeddings` in [05-rag-and-embeddings.md](05-rag-and-embeddings.md).)

---

## LangChain V1 architecture

```
              YOUR APP   (chatbot · RAG Q&A · agent …)
                  ▲  built with
   ┌──────────────┴──────────────────────────────────┐
   │ Chains & Agents              (langchain)         │  ← orchestration
   │   LCEL pipelines:  prompt | model | parser       │
   ├──────────────────────────────────────────────────┤
   │ Runnables + LCEL             (langchain-core)     │  ← foundation
   │   one interface: invoke · batch · stream          │
   ├──────────────────────────────────────────────────┤
   │ Model & tool integrations                        │  ← plug-ins
   │   langchain-openai · -anthropic · -community     │
   └──────────────────────────────────────────────────┘
                  ▲  talks to
        LLM providers · vector stores · tools · APIs
```

**How to read it (bottom → top):** integrations connect to the outside world (models, vector stores, tools); **`langchain-core`** wraps every piece as a **Runnable** with the same `invoke/batch/stream` interface; you compose those with **LCEL** into chains & agents; and those power your app. Each layer builds on the one below.

**Runnables — the core idea:**
- *Everything* is a runnable: prompts, models, output parsers, chains.
- Unified interface: `.invoke()` (one input) · `.batch()` (many) · `.stream()` (chunks).
- **LCEL:** chain runnables with the `|` pipe → concise, composable code. A chain is itself a runnable.

---

## Messages & roles

Instead of a string prompt, send a **list of messages with roles**:

| Role | LangChain class | Purpose |
|------|-----------------|---------|
| system | `SystemMessage` | Set behavior/constraints |
| user (human) | `HumanMessage` | The user's input |
| assistant (ai) | `AIMessage` | Model's reply (returned by invoke) |

```python
from langchain_core.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage("You are a helpful coding assistant. Answer only "
                  "programming questions; politely decline anything else."),
    HumanMessage("How do I reverse a list in Python?"),
]
response = llm.invoke(messages)   # -> AIMessage
```

- `AIMessage` has `.content` + metadata (token usage etc.).
- System message steers behavior: a programming question gets answered; "what is the meaning of life?" gets politely declined.

**Multi-turn conversation** = keep appending to the list:

```
[System, Human] ──invoke──▶ AIMessage
        └─ append AIMessage + new HumanMessage ──invoke──▶ next AIMessage ...
```

---

## LCEL basic chain (`core_concepts.py`)

The three components, piped together:

```
{"input": "..."} ──▶ prompt ──▶ model ──▶ parser ──▶ str
                 fills {input}   AIMessage    .content
                 → messages
```

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Respond in one sentence to the following input: {input}")
model  = ChatAnthropic(model="claude-haiku-4-5", temperature=0.7)
parser = StrOutputParser()

chain  = prompt | model | parser              # LCEL pipe = compose runnables
result = chain.invoke({"input": "What is the capital of France?"})  # -> str
```

**Key points:**
- `ChatPromptTemplate.from_template(...)` → single human message; `{vars}` interpolated at runtime.
- `chain.invoke()` takes a **dict keyed by template variable names** (all variables must be passed).
- `StrOutputParser` unwraps `AIMessage` → plain string.
- The chain is itself a runnable → also gets `.batch()` / `.stream()` for free.
- Any chat model works here (`ChatOpenAI`, `ChatAnthropic`, ...) — this repo uses `ChatAnthropic`.
- ⚠️ Model param needs the **ID** (`claude-haiku-4-5`), not the display name ("Claude Haiku 4.5").

---

## Batch execution — `.batch()`

Same chain, **list of input dicts** instead of one:

```python
chain = ChatPromptTemplate.from_template("Translate to Hindi: {input}") | model | parser

inputs  = [{"input": "Hello, how are you?"},
           {"input": "What is the weather like today?"},
           {"input": "Tell me a joke."}]
results = chain.batch(inputs)     # list[str], same order as inputs
```

```
[in1, in2, in3] ──▶ chain.batch() ──▶ [out1, out2, out3]
                    (runs in parallel)
```

- Every runnable gets `.batch()` free via the unified interface — no loop needed.
- Each dict must carry all template variables (`input` here).
- Results align 1:1 with inputs → pair with `zip(inputs, results)` or `enumerate`.

---

## Streaming — `.stream()`

Real-time token-by-token output (the ChatGPT/Claude typing effect). Same chain, third runnable method:

```python
chain = prompt | model | parser

for chunk in chain.stream({"input": "Explain quantum computing."}):
    print(chunk, end="", flush=True)   # chunks are str (thanks to parser)
```

```
model ──▶ "Quan" ▸ "tum " ▸ "comp" ▸ ... (yielded as generated)
```

- `.stream()` returns a **generator** — iterate, don't wait for the full response.
- `end="", flush=True` → chunks join seamlessly on one line, shown immediately.
- Parser applies per-chunk, so you get plain strings, not `AIMessageChunk`s.

### Runnable interface — recap

| Method | Input | Output | Use when |
|--------|-------|--------|----------|
| `.invoke()` | one dict | one result | single request |
| `.batch()` | list of dicts | list of results | many inputs, parallel |
| `.stream()` | one dict | generator of chunks | live/UX output |

---

## `init_chat_model` — provider-agnostic model init

Instead of importing a provider class (`ChatAnthropic`, `ChatOpenAI`, ...), use one universal factory:

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(model_provider="anthropic", model="claude-haiku-4-5",
                        temperature=0.7, max_tokens=1024)
# same thing, one string:  init_chat_model("anthropic:claude-haiku-4-5")
```

- Swap providers by changing **strings**, not imports → easy to make the model configurable (env var / config file).
- Returns a normal chat model runnable — drops straight into `prompt | model | parser`.

## Extras (`core_concepts.py`)

- **Multiple template variables** → all keys required at invoke:
  `from_template("...'{product_name}' targeting '{target_audience}'")` → `chain.invoke({"product_name": ..., "target_audience": ...})`
- **Schema inspection** — every runnable exposes its I/O contract:
  `chain.input_schema.model_json_schema()` / `chain.output_schema.model_json_schema()` (handy to see what a chain expects/returns).

---

*TODO (rest of section): few-shot prompting · structured LLM responses.*
