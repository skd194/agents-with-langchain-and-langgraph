# 1. Project Setup

> **Quick recall:** uv-managed Python project → `.env` holds API keys → `load_dotenv()` → `ChatAnthropic(...).invoke(...)` proves the connection works.

## Stack

- **uv** — package/env manager (`uv run main.py`, `uv add <pkg>`)
- **Python 3.11+** (pinned via `.python-version`)
- **Deps:** `langchain`, `langchain-core`, `langgraph`, `langchain-anthropic`, `langchain-openai`, `python-dotenv`

## API key flow

```
.env (ANTHROPIC_API_KEY / OPENAI_API_KEY)
        │ load_dotenv()
        ▼
ChatAnthropic(model="claude-haiku-4-5", temperature=0)
        │ .invoke("Say hello...")
        ▼
AIMessage(content=..., usage metadata)
```

- `.env.example` is the committed template; real keys stay in `.env` (git-ignored).
- `langchain_anthropic.ChatAnthropic` = LangChain chat-model wrapper; raw `anthropic.Anthropic` client also used (e.g. `client.models.list()` to inspect available models/capabilities).

## Local quirks (Windows)

- `sys.stdout.reconfigure(encoding="utf-8")` — console default cp1252 can't print emoji.
- `truststore.inject_into_ssl()` — Norton re-signs HTTPS; use OS cert store instead of certifi.

## Run

```bash
uv run main.py   # prints versions, model list, test LLM response
```
