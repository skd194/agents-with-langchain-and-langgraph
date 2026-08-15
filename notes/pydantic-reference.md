# Pydantic — reference

> **In one line:** Pydantic is a **data-validation** library. You declare the *shape* of data as a class (types + rules); it **enforces** that shape at runtime — checking, converting, and rejecting bad data automatically. It's "type hints that actually do something."

Used here for **structured LLM output** (`with_structured_output` in [06-rag-pipeline.md](06-rag-pipeline.md), `response_format` in [07-agents.md](07-agents.md)), but it's a general-purpose tool for any data boundary (APIs, configs, forms).

> This project runs **Pydantic v2** (bundled with LangChain 1.x). Methods below use v2 names (`model_dump`, `model_validate`, `model_json_schema`); v1's `.dict()` / `.parse_obj()` / `.schema()` are the old equivalents.

---

## Why it exists

Python type hints are **only documentation** — the runtime ignores them:

```python
def f(age: int): ...
f("hello")        # runs fine — "int" is never enforced
```

So any data crossing into your program (JSON, config, **LLM output**) needs manual checking:

```python
# without Pydantic — tedious and easy to get wrong
if "age" not in data:                raise ValueError("missing age")
if not isinstance(data["age"], int): raise TypeError("age not int")
if data["age"] < 0:                  raise ValueError("age negative")
```

Pydantic replaces all of that with a **declaration**:

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str
    age: int = Field(ge=0)      # must be >= 0
```

---

## The three things it does

**1. Validation** — bad data raises a clear error instead of silently corrupting state:

```python
User(name="Sam", age=30)       # ✅ User(name='Sam', age=30)
User(name="Sam", age=-5)       # ❌ ValidationError: Input should be >= 0
User(name="Sam", age="oops")   # ❌ ValidationError: not a valid integer
```

**2. Coercion** — converts when it safely can:

```python
User(name="Sam", age="30")     # "30" (str) → 30 (int), automatically
```

**3. Structured, typed access** — a real object, not a loose dict:

```python
u = User(name="Sam", age=30)
u.name                  # "Sam"   (attribute access, IDE autocomplete)
u.model_dump()          # {"name": "Sam", "age": 30}     → back to a dict
u.model_dump_json()     # '{"name":"Sam","age":30}'      → JSON string
u.model_json_schema()   # JSON Schema describing the class
```

---

## `BaseModel` — declaring a shape

Subclass `BaseModel`; each attribute is `name: type`. One class = one data shape.

```python
class Product(BaseModel):
    name: str
    price: float
    in_stock: bool = True          # default → optional
    tags: list[str] = []           # typed collection
```

- A field **with no default is required**; **with a default is optional**.
- Nesting works — a field can be another `BaseModel`:

```python
class Order(BaseModel):
    id: int
    customer: User                 # nested model
    items: list[Product]           # list of models

Order.model_validate(raw_dict)     # validates the whole tree at once
```

### Common field types

| Type | Meaning |
|------|---------|
| `str`, `int`, `float`, `bool` | scalars (coerced when safe) |
| `list[str]`, `dict[str, int]` | typed collections |
| `Optional[str]` / `str \| None` | may be `None` |
| `Literal["high","medium","low"]` | must be one of a fixed set |
| `datetime`, `UUID`, `EmailStr`* | rich types (`EmailStr` needs `pydantic[email]`) |
| another `BaseModel` | nested object |

---

## `Field` — rules & metadata on one attribute

`Field(...)` annotates a single field beyond its bare type: a description, a default, and validation constraints.

```python
class Account(BaseModel):
    username: str  = Field(min_length=3, max_length=20)
    age: int       = Field(ge=0, le=120)                 # 0..120
    email: str     = Field(description="primary email")  # doc/instruction
    role: str      = Field(default="user")               # default value
    user_id: int   = Field(alias="id")                   # JSON key is "id"
```

| Argument | Purpose | Example |
|----------|---------|---------|
| `description` | human/LLM-readable doc (goes into the schema) | `Field(description="...")` |
| `default` / `default_factory` | value if omitted | `Field(default=[])`, `Field(default_factory=list)` |
| `ge` / `le` / `gt` / `lt` | numeric bounds | `Field(ge=0, le=1)` |
| `min_length` / `max_length` | string / list length | `Field(min_length=1)` |
| `pattern` | regex the string must match | `Field(pattern=r"^\d{4}$")` |
| `alias` | external JSON key name | `Field(alias="user_id")` |

> Use `default_factory=list` (not `default=[]`) for mutable defaults — it makes a fresh list per instance.

---

## Validators — custom rules

When a constraint needs logic, use a validator:

```python
from pydantic import BaseModel, field_validator

class Signup(BaseModel):
    password: str
    password_confirm: str

    @field_validator("password")
    @classmethod
    def strong_enough(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("password too short")
        return v
```

- `@field_validator("field")` runs per-field; `@model_validator` runs on the whole object (e.g. cross-field checks like "passwords match").
- Raise `ValueError` to reject; return the (possibly cleaned) value to accept.

---

## Loading data in / out

```python
# IN — from a dict or JSON (validates)
User.model_validate({"name": "Sam", "age": 30})        # from dict
User.model_validate_json('{"name":"Sam","age":30}')    # from JSON string

# OUT — to a dict or JSON
u.model_dump()            # dict
u.model_dump_json()       # JSON string
u.model_dump(exclude={"age"})          # drop fields
u.model_dump(exclude_none=True)        # skip None values
```

| Direction | v2 method | v1 (old) |
|-----------|-----------|----------|
| dict → model | `model_validate` | `parse_obj` |
| JSON → model | `model_validate_json` | `parse_raw` |
| model → dict | `model_dump` | `dict` |
| model → JSON | `model_dump_json` | `json` |
| model → schema | `model_json_schema` | `schema` |

---

## Why LangChain leans on Pydantic

The key method is `model_json_schema()` — Pydantic turns your class into a **JSON Schema** (a language-neutral description of the shape). LangChain hands that to the LLM:

```
YourModel (Pydantic class)
     │  model_json_schema()
     ▼
JSON Schema ──▶ sent to the LLM as "produce data in THIS shape"
     ▼
LLM returns JSON ──▶ Pydantic VALIDATES it ──▶ a real typed object
```

So in `llm.with_structured_output(Model)` Pydantic does **double duty**:

1. **Before the call** — its schema (field names, types, and every `Field(description=...)`) tells the model exactly what to produce. *This is why descriptions matter — they travel into the prompt the model sees.*
2. **After the call** — it validates the returned JSON into an instance, guaranteeing the fields exist and match types. No manual parsing, no `KeyError`.

### As used in this project (`013_rag_pipeline.py`)

```python
from pydantic import BaseModel, Field
from typing import List

class RAGResponse(BaseModel):
    """Structured RAG response."""
    answer: str            = Field(description="The answer to the question")
    confidence: str        = Field(description="high, medium, or low")
    sources_used: List[str]= Field(description="List of sources referenced")
    follow_up: str         = Field(description="Suggested follow-up question")

structured_llm = llm.with_structured_output(RAGResponse)

result = rag_chain.invoke("What is LangGraph?")
result.answer         # -> str    (typed attribute, no parsing)
result.confidence     # -> "high"
result.sources_used   # -> ["langchain_knowledge_base.md"]
```

- The class *is* the contract: the chain ends at `structured_llm` — **no `StrOutputParser`**, because the output is an object, not text.
- The same schema mechanism powers an agent's `response_format=Model` in [07-agents.md](07-agents.md).

> **Mental model:** Pydantic is the **contract enforcer** at your program's boundary. Anywhere messy external data enters — API, config, or **LLM output** — a `BaseModel` guarantees that past that line the data is exactly the shape your code expects.
