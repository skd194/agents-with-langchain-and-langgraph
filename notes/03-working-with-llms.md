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
