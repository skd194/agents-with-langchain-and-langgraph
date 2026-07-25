# 3. Working with LLMs

> **Quick recall:** Configure models via `init_chat_model` (provider-agnostic). Talk to them with **message objects** — `SystemMessage` / `HumanMessage` in, `AIMessage` out. Multi-turn = keep appending to the message list; the model is stateless, *you* carry the history.

**Covers:** provider setup & configuration · message objects · multi-turn conversation · comparing models

---

## Model configuration

`init_chat_model` accepts the full set of runtime knobs (`working_with_llms.py`):

```python
model = init_chat_model(
    model_provider="anthropic",     # inferred from model name if omitted
    model="claude-haiku-4-5",
    temperature=0.7,                # creativity (0 = deterministic)
    max_tokens=1024,                # response length cap → cost control
    timeout=60,                     # seconds before request aborts
    max_retries=3,                  # auto-retry on transient failures
)
```

- Swapping providers = swap strings; the rest of the code is untouched.
- Guard optional providers: `if os.getenv("ANTHROPIC_API_KEY"): ...` before init.
- **Compare models:** keep a `dict[name, model]`, loop the same prompt through each, print responses side by side.

## Message anatomy

Messages are the fundamental unit of context — **role + content + metadata**. Printing one reveals more than text:

```
SystemMessage / HumanMessage / AIMessage
├─ content            "You are an engineer..."
├─ additional_kwargs  {}
└─ response_metadata  {token usage, model id, stop reason, ...}  ← rich on AIMessage
```

Wrapper classes (vs raw strings) let LangChain attach and carry this metadata through the workflow — increasingly valuable as the app grows.

## Multi-turn conversation

```python
messages = [
    SystemMessage(content="You are an engineer, always answer like a software engineer."),
    HumanMessage(content="What is the capital of France?"),
]
response = model.invoke(messages)                    # -> AIMessage

messages.append(response)                            # keep the model's turn
messages.append(HumanMessage(content="What is the capital of India?"))
followup = model.invoke(messages)                    # answers with full context
```

```
[System, Human] ─invoke─▶ AI ─append─▶ [S, H, AI, Human₂] ─invoke─▶ AI₂ ...
```

- The system message persists across turns — it grounds every response.
- Appended `AIMessage`s carry their token-usage metadata into the history.
- This list *is* the conversation memory (in-process only — persistence comes later).

---

## Prompt templates

A template is a **cookie-cutter**: fixed text + `{variables}` filled at runtime → reusable, define once, use with any values.

```
"Tell me a {adjective} joke about {topic}"  +  {adjective:"funny", topic:"cats"}
                                            ▼
              HumanMessage("Tell me a funny joke about cats")
```

**Multi-message template** — ground the system role *and* take user input in one structure:

```
System: "You are a {role}, always be {tone}"   {role:"tutor", tone:"encouraging",
Human:  "{question}"                             question:"explain recursion"}
                          ▼
System: "You are a tutor, always be encouraging"
Human:  "explain recursion"
```

Benefits: **encapsulated** (whole prompt in one object) · **reusable** · **modular**.

**In code** (`003_prompt_messages.py`) — `.format_messages(**vars)` fills the template and returns ready-to-invoke message objects:

```python
# Single-message template → LangChain infers a HumanMessage
prompt = ChatPromptTemplate.from_template("Tell me a {adjective} story about {topic}")
msgs   = prompt.format_messages(adjective="funny", topic="cat")
# [HumanMessage(content="Tell me a funny story about cat")]

# Multi-message template — tuple shorthand ("role", "text with {vars}")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
    ("human",  "Translate the following text: {text}"),
])
msgs     = prompt.format_messages(input_language="English", output_language="Hindi", text="How are you?")
response = model.invoke(msgs)
```

- `("system", ...)` / `("human", ...)` / `("ai", ...)` tuples are shorthand for the message classes — no need to instantiate them yourself.
- Template variables must be supplied at `format_messages` time (all of them).

## Message types

| Type | Purpose |
|------|---------|
| `SystemMessage` | Sets behavior / persona — grounds *who the AI is* |
| `HumanMessage` | User input — the query or task |
| `AIMessage` | The model's response (fed back in for multi-turn) |
| `ToolMessage` | Wraps results from tools / APIs / DBs / functions |

Typical flow: `System → Human → AI → Human → AI → ...` (with `Tool` messages woven in when tools are used).

```python
from langchain_core.messages import (
    HumanMessage, SystemMessage, AIMessage, ChatMessage, ToolMessage,
)
# any mix can go into a single messages list passed to the model
```

## MessagesPlaceholder — dynamic history

A named **slot** in a template where a variable-length list of prior messages gets injected at runtime — the clean way to carry conversation history into a prompt.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),   # ← past turns drop in here
    ("human", "{question}"),
])

history = [                                          # earlier conversation
    HumanMessage(content="Hey, my name is Paulo"),
    AIMessage(content="Nice to meet you, Paulo"),
]
msgs = prompt.format_messages(history=history, question="What is my name?")
# model answers "Paulo" — the placeholder gave it the context
```

```
[ system ] [ …history… ] [ human question ]
              ▲ list injected here at format time
```

- `variable_name` is the key you pass to `format_messages` (a list of message objects).
- Length is flexible — 0 to many prior turns — unlike fixed `("human", ...)` slots.
- This is what makes memory/chat-history integrations plug in cleanly later.

## Few-shot prompting

Teach by **example** — show the pattern, the model generalizes it. No fine-tuning; just examples in the prompt.

```
happy → sad          (examples establish "give the opposite")
tall  → short
hot   → cold
─────────────
fast  → ?   ⇒  model answers "slow"
```

- **2–5 examples** are usually enough to lock in a pattern.

**In code** — `FewShotChatMessagePromptTemplate` turns a list of examples into message pairs, then embeds into the final prompt (`003_prompt_messages.py`, sentiment classifier):

```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples = [
    {"input": "I absolutely love this phone.", "output": "Positive"},
    {"input": "The service was terrible.",     "output": "Negative"},
    {"input": "The package arrived yesterday.", "output": "Neutral"},
]

example_prompt  = ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")])
few_shot_prompt = FewShotChatMessagePromptTemplate(examples=examples, example_prompt=example_prompt)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify sentiment as Positive / Negative / Neutral. Return ONLY the label."),
    few_shot_prompt,                        # examples expand here as human/ai turns
    ("human", "{input}"),
])

msgs     = final_prompt.format_messages(input="The movie was fantastic.")
response = model.invoke(msgs)               # -> "Positive"
```

- `example_prompt` defines the shape of *one* example (human input → ai output).
- The few-shot block drops straight into `from_messages` between system and the live human turn.
- Pair with `temperature=0` and a small `max_tokens` for crisp, deterministic labels.

## Prompt composition

Build complex prompts from **reusable building blocks**:

```
system_prompt ("You are a {role}")  +  user_prompt ("{question}")  →  full_prompt
```

- **Reusability** — same blocks across projects.
- **Modularity** — swap a component without touching the rest.
- **Maintainability** — update once, applies everywhere.

Seen in the few-shot example above: `system` + `few_shot_prompt` + `human` are three independent blocks composed into one `final_prompt`.

---

## Chains with LCEL

**LCEL** (LangChain Expression Language) is the current standard. The legacy `LLMChain` / `SequentialChain` / `TransformChain` classes are **deprecated** — don't use them.

```python
chain = prompt | model | parser      # a RunnableSequence
```

- The `|` pipe builds a **RunnableSequence**; data flows **left → right**.
- Each step's output feeds the next: `input → prompt.invoke → model.invoke → parser.invoke → output`.
- The chain is itself a runnable, composed of runnables → **composable all the way down**.

## Composition patterns — in detail

All examples from `005_chains_v1.py`. The building blocks:

| Runnable | What it does | Output |
|----------|--------------|--------|
| `RunnableSequence` (the `\|` pipe) | Runs steps left→right, each output feeds the next | Output of the **last** step |
| `RunnableParallel` | Runs several branches on the **same input** concurrently | A **dict** — one key per branch |
| `RunnableLambda` | Wraps a plain Python function so it can live inside a chain | Whatever your function returns |
| `RunnablePassthrough` | Identity — forwards its input **unchanged** | Its input, as-is |
| `RunnableBranch` | Conditional routing — runs the first chain whose condition is `True`, else a default | Output of the chosen chain |

### What is `RunnableLambda`?

**A wrapper that turns any ordinary Python function into a runnable**, so your own logic can sit inside an LCEL pipe alongside prompts, models, and parsers.

**Why it's needed:** the `|` pipe only connects *runnables*. A plain function (`def f(x): ...` or a `lambda`) is **not** a runnable, so you can't drop it into a chain directly. `RunnableLambda(f)` adapts it — now it has `.invoke()` / `.batch()` / `.stream()` like everything else.

**How it works:**

```
prev step's output ──▶ RunnableLambda(f) ──▶ f(output) ──▶ next step
```

- The wrapped function takes **exactly one argument** — whatever the previous step produced.
- Whatever it **returns** becomes the input to the next step.
- No decorators, no subclassing — just wrap and pipe.

```python
RunnableLambda(lambda x: x.upper())     # input "hi"  -> output "HI"
RunnableLambda(fake_retriever)          # input dict  -> whatever the function returns
```

**Two typical jobs** (both seen in the pass-through example below):
1. **Call your own code inside a chain** — e.g. `RunnableLambda(fake_retriever)` runs a retriever function as a step.
2. **Reshape data between steps** — e.g. `RunnableLambda(lambda inputs: {...})` flattens/renames keys so the next runnable (a prompt) gets exactly the variables it expects.

---

### 1. Basic (sequence) chain

```python
prompt = ChatPromptTemplate.from_template("Summarize the following text in one sentence: {text}")
parser = StrOutputParser()
chain  = prompt | model | parser
result = chain.invoke({"text": "LangChain is a framework ..."})   # -> str
```

**What actually happens** (data flows left → right):

```
{"text": ...}
   │  prompt.invoke  → fills template → PromptValue (a list of messages)
   ▼
   model.invoke      → sends messages to LLM → AIMessage
   ▼
   parser.invoke     → unwraps AIMessage.content → plain string
```

Each step's output is the next step's input. The whole `chain` is itself a runnable → it also has `.batch()` / `.stream()`.

---

### 2. Parallel chain — `RunnableParallel`

Run several independent chains on the **same input** at once.

```python
from langchain_core.runnables import RunnableParallel

analysis_chain = RunnableParallel(
    summary   = summarize_prompt | model | parser,
    sentiment = sentiment_prompt | model | parser,
    keywords  = keywords_prompt  | model | parser,
)
result = analysis_chain.invoke({"text": text})
# result -> {"summary": "...", "sentiment": "...", "keywords": "..."}
for key, value in result.items():
    print(key, value)
```

**What actually happens:**

```
                 ┌─▶ summary branch   (prompt|model|parser) ─┐
{"text": text} ──┼─▶ sentiment branch (prompt|model|parser) ─┼─▶ {"summary":..., "sentiment":..., "keywords":...}
                 └─▶ keywords branch  (prompt|model|parser) ─┘
                     (all three run concurrently)
```

- **Output = a dict**, keys = the argument names (`summary`, `sentiment`, `keywords`); values = each branch's result.
- Every branch receives the **same** input dict (`{"text": text}`).
- Concurrent, not sequential → faster than three separate `.invoke()` calls.
- `analysis_chain` is itself a runnable → nest it inside larger chains freely.

---

### 3. Pass-through chain — RAG shape (`RunnablePassthrough` + `RunnableLambda`)

The pattern for keeping the **original question** while also attaching **retrieved context**.

```python
prompt = ChatPromptTemplate.from_template(
    "Original question: {question}\nContext: {context}\nAnswer the question based on the context.")

def fake_retriever(input_dict):                 # stands in for a real vector-store retriever
    return "LangChain was created by Harrison Chase in 2022."

chain = (
    RunnableParallel(
        context  = RunnableLambda(fake_retriever),   # fetch context
        question = RunnablePassthrough(),            # forward input unchanged
    )
    | RunnableLambda(lambda inputs: {                # reshape into flat prompt vars
        "question": inputs["question"]["question"],
        "context":  inputs["context"],
      })
    | prompt
    | model
    | StrOutputParser()
)
result = chain.invoke({"question": "Who created LangChain?"})
```

**Step-by-step — what each stage produces:**

```
invoke {"question": "Who created LangChain?"}

① RunnableParallel
   context  = fake_retriever(input) = "LangChain was created by ..."
   question = Passthrough()         = {"question": "Who created LangChain?"}   ← whole input!
   ⇒ {"context": "...", "question": {"question": "Who created LangChain?"}}

② RunnableLambda (flatten)
   ⇒ {"question": "Who created LangChain?", "context": "..."}

③ prompt  → fills {question} & {context} → messages
④ model   → AIMessage
⑤ parser  → "LangChain was created by Harrison Chase in 2022."
```

- **What is the "input dict"?** It's whatever you pass to `chain.invoke(...)` — here `{"question": "Who created LangChain?"}`. That same dict is handed to **every branch** of the `RunnableParallel` as their argument:
  - `fake_retriever(input_dict)` receives `{"question": "..."}` (it ignores it and returns fixed text; a real retriever would read `input_dict["question"]` to search).
  - `RunnablePassthrough()` receives the same `{"question": "..."}` and returns it untouched.
  - So the parameter name `input_dict` is just the branch's view of the original `.invoke()` payload — a plain Python `dict`, not anything LangChain-specific.
- **Why the nested `inputs["question"]["question"]`?** `RunnablePassthrough` forwards the *entire* input dict, so under key `question` you get `{"question": ...}` — the lambda flattens it back out.
- **Why the pattern matters:** a real retriever turns the query into *documents* and loses the original wording. Pass-through preserves the question so the prompt gets **both** the context and the real question.
- Everything is composable — swap `fake_retriever` for a real one, or add more branches, without changing the shape.

---

### 4. Branching chain — `RunnableBranch`

Conditional routing: pick which sub-chain runs based on the input (like `if / elif / else`).

```python
from langchain_core.runnables import RunnableBranch

# Dedicated classifier model at temperature=0 → deterministic label (not the 0.7 answer model)
classifier_model = init_chat_model(model="gpt-4o-mini", model_provider="openai", temperature=0)
classifier = classifier_prompt | classifier_model | StrOutputParser()   # returns "code" or "general"

def is_code_question(input_dic):                 # a condition = function returning True/False
    classification = classifier.invoke(input_dic)
    return "code" in classification.lower()

branch = RunnableBranch(
    (is_code_question,                           # (condition, chain)
     code_prompt | model | StrOutputParser()
     | RunnableLambda(lambda x: {"branch": "code", "answer": x})),
    general_prompt | model | StrOutputParser()   # default (last arg, no condition)
     | RunnableLambda(lambda x: {"branch": "general", "answer": x}),
)

result = branch.invoke({"input": "How do I write a for loop in Python?"})
print(result["branch"], result["answer"])        # -> "code", "Certainly! In Python..."
```

**What actually happens:**

```
{"input": q}
   │
   ▼  is_code_question(input)?  ── True ──▶ code_prompt    | model | parser | tag("code")
   │                             └ False ─▶ general_prompt | model | parser | tag("general")
   ▼
 {"branch": <route>, "answer": <str>}
```

- `RunnableBranch` takes `(condition, runnable)` pairs, then a final **default** runnable with no condition.
- Conditions are checked **top to bottom**; the first `True` wins and its chain runs. If none match → default.
- Each condition is a function `input -> bool`; here it calls the LLM classifier to decide the route.
- **Deterministic routing:** the classifier uses its own `temperature=0` model so the label is stable, while the answer model can stay creative (`0.7`).
- **Tag the route:** a trailing `RunnableLambda` wraps each branch's output into `{"branch": ..., "answer": ...}`, so the caller can see *which* path ran, not just the text.
- **Cost note:** this pattern makes **two LLM calls per request** — ① the classifier (picks the route) + ② the selected chain (produces the answer).

## Debugging chains

**High-level levels:**

| Level | How | Use |
|-------|-----|-----|
| **Logging** | `set_debug(True)` | Quick step-by-step prints in the console |
| **Callbacks** | custom handler classes | Intercept & log each step; flexible/customizable |
| **LangSmith** | tracing platform | Full observability — traces, cost, evaluation (recommended) |

**Practical methods** (`005_chains_v1.py`, `demo_debbuging`):

### Method 1 — inspect the schema

See exactly what a chain expects as input and produces as output — verify its interface before wiring it up.

```python
chain = prompt | model | StrOutputParser()

print(chain.input_schema.model_json_schema())    # e.g. {name: str}  (from the prompt)
print(chain.output_schema.model_json_schema())   # StrOutputParserOutput -> string
```

- `.model_json_schema()` gives the **full** JSON schema (fields, types, required); the bare `input_schema` object shows just the type.
- Tells you what types flow in/out — the first thing to check when a chain errors on integration.

### Method 2 — `with_config` for tracing

Attach a run name (and tags/metadata) so the execution is identifiable in logs/LangSmith.

```python
result = chain.with_config(run_name="greeting_chain").invoke({"name": "Alice"})
```

- `with_config(...)` returns the same chain with metadata attached — doesn't change behavior.
- Useful for labelling runs and passing tags/metadata through to observability tooling.

### Method 3 — inspect intermediate steps (the "tap")

Insert `RunnableLambda` loggers *between* steps to see the data at each stage.

```python
def log_step(x, step_name=""):
    print(f"[{step_name}] {type(x).__name__}: {str(x)[:100]}")
    return x                                  # ← MUST return x unchanged

debug_chain = (
    prompt
    | RunnableLambda(lambda x: log_step(x, "after_prompt"))   # logs PromptValue
    | model
    | RunnableLambda(lambda x: log_step(x, "after_model"))    # logs AIMessage
    | StrOutputParser()
)
debug_chain.invoke({"name": "Debug"})
```

```
prompt ─▶ [tap: after_prompt] ─▶ model ─▶ [tap: after_model] ─▶ parser
              logs & forwards              logs & forwards
```

- A **tap** = sees the data, logs it, **passes it through unchanged**.
- `return x` is essential — the lambda must forward the value so the next step still receives it; without it the chain breaks.
- Lets you watch each transformation (e.g. `after_prompt` → `ChatPromptValue`, `after_model` → `AIMessage`).
