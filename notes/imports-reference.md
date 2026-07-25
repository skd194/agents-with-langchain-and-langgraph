# Imports Reference

Every import used across the project's Python files, why it's needed, and a short example. Grouped by source. Each block is copy-pastable.

> Keep this in sync whenever imports change in the `.py` files.
>
> _Code-block background follows your Markdown preview theme — in VS Code, switch **Color Theme** (or a lighter preview theme) if the default looks too dark._

## Standard library

Reconfigure `sys.stdout` to UTF-8 so emoji in model output don't crash the Windows cp1252 console.
```python
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
```

Read an installed package's version at runtime.
```python
from importlib.metadata import version
version("langchain-core")   # -> '0.3.x'
```

Operating-system utilities (e.g. delete a temp file after loading).
```python
import os
os.remove(temp_file_path)
```

Create temporary files to demo loaders without shipping fixtures.
```python
import tempfile
tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
```

Filesystem path handling.
```python
from pathlib import Path
Path("docs/report.pdf")
```

## Environment & TLS

Load API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) from `.env` into the environment.
```python
from dotenv import load_dotenv
load_dotenv()
```

Verify TLS via the Windows certificate store (Norton re-signs HTTPS, so certifi fails).
```python
import truststore
truststore.inject_into_ssl()
```

## Model providers

LangChain chat-model wrapper for Claude; invoke or pipe into chains.
```python
from langchain_anthropic import ChatAnthropic
ChatAnthropic(model="claude-haiku-4-5", temperature=0)
```

Raw Anthropic SDK client; list models & inspect capabilities.
```python
from anthropic import Anthropic
Anthropic().models.list()
```

Provider-agnostic factory; pick model/provider by string.
```python
from langchain.chat_models import init_chat_model
init_chat_model("anthropic:claude-haiku-4-5")
```

## Prompts

Role-based prompt templates with `{variables}`.
```python
from langchain_core.prompts import ChatPromptTemplate
ChatPromptTemplate.from_template("Hi {name}")
```

Turn examples into human/ai pairs for few-shot prompting.
```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate
FewShotChatMessagePromptTemplate(examples=ex, example_prompt=p)
```

Named slot for injecting variable-length conversation history.
```python
from langchain_core.prompts import MessagesPlaceholder
MessagesPlaceholder(variable_name="history")
```

## Output parsers

Unwrap the model's `AIMessage` into a plain string.
```python
from langchain_core.output_parsers import StrOutputParser
chain = prompt | model | StrOutputParser()
```

## Messages

Wrap user input with the `human` role.
```python
from langchain_core.messages import HumanMessage
HumanMessage(content="What is LangChain?")
```

Set behavior/persona with the `system` role.
```python
from langchain_core.messages import SystemMessage
SystemMessage(content="You are a tutor")
```

The model's reply; append to history for multi-turn.
```python
from langchain_core.messages import AIMessage
messages.append(AIMessage(content="Hi!"))
```

Message with an arbitrary custom role.
```python
from langchain_core.messages import ChatMessage
ChatMessage(role="critic", content="...")
```

Wrap results returned from tools / functions / APIs.
```python
from langchain_core.messages import ToolMessage
ToolMessage(content="42", tool_call_id="abc")
```

Module-level import of the messages package.
```python
from langchain_core import messages
messages.HumanMessage(content="Hi")
```

## Runnables

Run branches on the same input concurrently → dict of results.
```python
from langchain_core.runnables import RunnableParallel
RunnableParallel(summary=chain1, keywords=chain2)
```

Identity runnable; forward input unchanged (preserve original query).
```python
from langchain_core.runnables import RunnablePassthrough
{"question": RunnablePassthrough()}
```

Wrap a plain function so it fits in an LCEL pipe (tap / transform).
```python
from langchain_core.runnables import RunnableLambda
RunnableLambda(lambda x: x.upper())
```

Conditional routing; first `True` condition wins, else default.
```python
from langchain_core.runnables import RunnableBranch
RunnableBranch((is_code, code_chain), default_chain)
```

## Documents & loaders

The core `Document` type (`page_content` + `metadata`); usually produced by loaders, can be built by hand.
```python
from langchain_core.documents import Document
Document(page_content="...", metadata={"source": "a.txt"})
```

Load a plain-text file into `Document` objects.
```python
from langchain_community.document_loaders import TextLoader
TextLoader("notes.txt").load()   # -> list[Document]
```

Scrape web page(s) into `Document` objects (needs BeautifulSoup).
```python
from langchain_community.document_loaders import WebBaseLoader
WebBaseLoader("https://example.com").load()
```

Load many files from a folder in bulk.
```python
from langchain_community.document_loaders import DirectoryLoader
DirectoryLoader("docs/", glob="**/*.txt", loader_cls=TextLoader).load()
```

Load a PDF into per-page `Document` objects (needs `uv add pypdf`).
```python
from langchain_community.document_loaders import PyPDFLoader
PyPDFLoader("report.pdf").load()
```

HTML/XML parser backing `WebBaseLoader`; install via `uv add bs4`.
```python
from bs4 import BeautifulSoup
BeautifulSoup(html, "html.parser")
```

## Embeddings

Turn text into vectors via OpenAI embedding models.
```python
from langchain_openai.embeddings import OpenAIEmbeddings
OpenAIEmbeddings(model="text-embedding-3-small").embed_query("hi")
```

---

### Known issues

- `010_embeddings.py:29` — `from langchain_community. import Embeddings` is **incomplete** (trailing dot) and raises `SyntaxError`, blocking the whole file. Intended target is likely `from langchain_community.embeddings import HuggingFaceEmbeddings` (local, free embeddings).
