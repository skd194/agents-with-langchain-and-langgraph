# 7. Agents (primer)

> **Quick recall:** A **chain** runs fixed steps *you* define. An **agent** lets the **LLM decide** what to do next — which tool to call, and whether to keep going — running in a **loop** until it has an answer. In LangChain 1.x you build one with `create_agent` (built on LangGraph).

> _Primer — concepts now; expand with real code as the course's agents/LangGraph section is covered._

---

## Chain vs Agent

```
Chain:  input → step1 → step2 → step3 → output      (you decide the path)
Agent:  input → LLM ⇄ tools ⇄ LLM ⇄ tools → answer   (the LLM decides the path)
```

- **Chain** — known, linear steps (`prompt | model | parser`).
- **Agent** — the model chooses the steps dynamically; may call tools and loop.

## The agent loop (ReAct)

**ReAct = Reasoning + Acting.** The **name** comes from just those two words — but the operating cycle has **three** parts: **Thought → Action → Observation**. *Observation* isn't in the acronym; it's simply the **feedback you get back from acting** (the tool's reply), and it's what lets the model reason again. So: **Act** produces the **Observation**, and the **Observation** feeds the next **Thought**.

The LLM interleaves **thinking** and **doing** instead of answering in one shot — it reasons about the next step, acts (calls a tool), observes the result, and repeats:

```
Thought      → reason about what to do next
Action       → call a tool (with an input)
Observation  → read the tool's result
   ↑____________________________________|
        repeat until it can answer
Final Answer → respond to the user
```

- Reasoning alone → can plan but can't fetch facts (may hallucinate).
- Acting alone → can call tools but doesn't know which/when.
- **ReAct** combines both: reasoning picks the actions; observations ground the reasoning.

**Example trace** — *"What's the weather where the Eiffel Tower is?"*

```
Thought: The Eiffel Tower is in Paris → I need Paris's weather.
Action:  check_weather("Paris")
Observation: "18°C, cloudy"
Thought: I have the answer.
Final Answer: It's 18°C and cloudy in Paris.
```

**Multi-step trace** — the agent **reasons again after each observation**, chaining tools.
*"How much for 3 nights at the top-rated hotel in Paris?"*

```
Thought: First I need the top-rated hotel in Paris.
Action:  search_hotels("Paris")
Observation: "Top-rated: Hotel Lumière — $220/night"
Thought: Got the price ($220/night). Now I need 3 × 220.
Action:  calculator("220 * 3")
Observation: "660"
Thought: I now have the total.
Final Answer: 3 nights at Hotel Lumière would cost $660.
```

Notice each **Observation feeds the next Thought** — the agent didn't know the second step (the multiply) until the first tool told it the nightly price.

**Maps to LangChain messages** (from [03-working-with-llms.md](03-working-with-llms.md)):

| ReAct step | LangChain message |
|------------|-------------------|
| Thought + Action | `AIMessage` with `tool_calls` |
| Observation | `ToolMessage` (tool result) |
| Final Answer | `AIMessage` with plain content (no tool calls) |

So the loop is:

```
1. LLM gets the messages (+ system prompt)
2. Wants a tool? → AIMessage with tool_calls
3. Tools run     → results appended as ToolMessage(s)
4. LLM called again with the new info
5. Repeat 2–4 until a normal answer (no tool_calls)
```

> Agent = **LLM + tools + a loop that runs until done.** (Modern tool-calling models do the "Thought" implicitly via native tool-calling, not literal `Thought:` text — but the reason→act→observe cycle is the same.)

## Tools

A tool is a function the agent may call; its **docstring + type hints** tell the LLM what it does:

```python
def check_weather(location: str) -> str:
    """Return the weather forecast for the specified location."""   # LLM reads this
    return f"It's always sunny in {location}"
```

Common tools: web search, calculators, API calls, DB queries, code execution, a **retriever** (RAG-as-a-tool).

## Creating an agent (LangChain 1.x)

### What `create_agent` does

A **factory** that wires up the whole ReAct loop for you and returns a ready-to-run agent (a compiled LangGraph graph). You give it three main things:

| Arg | What it is |
|-----|-----------|
| `model` | the LLM (string like `"anthropic:claude-sonnet-4-5"` or a chat model) |
| `tools` | list of functions the agent may call |
| `system_prompt` | instructions / persona for the agent |

It handles everything in between: call the model → if it requests tools, run them → feed results back → repeat until a final answer. You **don't write the loop yourself**.

```python
from langchain.agents import create_agent

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[check_weather],
    system_prompt="You are a helpful assistant",
)
result = agent.invoke({"messages": [{"role": "user", "content": "weather in SF?"}]})
```

### Basic behavior

**Input/output is a messages list** (the running conversation), not a plain string:

```python
# invoke → returns the full messages list; last message is the answer
out = agent.invoke({"messages": [{"role": "user", "content": "weather in SF?"}]})
print(out["messages"][-1].content)      # -> "It's always sunny in SF"

# stream → watch each step (model call, tool call, result) as it happens
for chunk in agent.stream({"messages": [{"role": "user", "content": "weather in SF?"}]},
                          stream_mode="updates"):
    print(chunk)
```

What happens on that call:
```
user "weather in SF?"
  → model decides to call check_weather("SF")     (AIMessage + tool_calls)
  → tool runs → "It's always sunny in SF"          (ToolMessage)
  → model reads it, returns final answer            (AIMessage)
```

- **No tools?** `create_agent(model, tools=[])` behaves like a plain chat model.
- **Multiple tools?** The model picks which to call (and can call several, in sequence) based on the question.
- Built on **LangGraph** → the loop is a stateful graph with cycles.
- Lower-level equivalent: `create_react_agent` from `langgraph.prebuilt`; for full control, build a `StateGraph` yourself.

## The loop in pseudocode (multiple iterations)

What `create_agent` runs internally is essentially a `while` loop that keeps going until the model stops asking for tools:

```python
messages = [system_prompt, user_question]

while True:
    ai = model.invoke(messages)          # model's turn
    messages.append(ai)

    if not ai.tool_calls:                # no tool requested → we're done
        break

    for call in ai.tool_calls:           # run each requested tool
        result = run_tool(call)
        messages.append(ToolMessage(result))

return messages                          # last message = final answer
```

**Multi-iteration trace** — *"Is it warmer in Paris or Tokyo?"* (needs two tool calls):

```
iter 1:  AIMessage  → tool_calls: check_weather("Paris")
         ToolMessage → "18°C"
iter 2:  AIMessage  → tool_calls: check_weather("Tokyo")
         ToolMessage → "25°C"
iter 3:  AIMessage  → "Tokyo is warmer — 25°C vs 18°C."   (no tool_calls → loop ends)
```

- Each **iteration** = one model call. The loop ran **3 times**: two to gather data, one to answer.
- The agent decides *how many* iterations it needs — you don't set it (though you can cap it to avoid runaway loops).

## Why agents

Decide-and-act apps a linear chain can't do: research assistants, tool-using copilots, multi-step automation, and (with LangGraph) **multi-agent** systems and human-in-the-loop approval.

> **Rule of thumb:** known steps → **chain** (LCEL); steps depend on the model's decisions / may loop → **agent** (`create_agent` / LangGraph).

## If `create_agent` does it all, why still need LangChain/LangGraph?

Because `create_agent` **is** LangChain (built on LangGraph) — a convenience factory, not a replacement. It covers **one pattern well**; you need the rest of the toolkit around it.

```
        create_agent          ← easy button: ONE prebuilt agent (the ReAct loop)
             │ built on
   ┌─────────┴──────────┐
   │  LangChain / LCEL  │     ← building blocks (models, prompts, RAG, tools, parsers)
   │  + LangGraph       │       AND custom chains & graphs
   └────────────────────┘
```

1. **It *is* the framework** — calling `create_agent` means you're already using LangChain + LangGraph.
2. **It only handles agents** — the deterministic majority (summarize, classify, RAG Q&A with fixed steps) is cheaper and more predictable as a plain **LCEL chain**, not an autonomous loop.
3. **It can't build the parts it uses** — the model, prompts, and especially **tools** (e.g. a retriever = your whole RAG stack from notes 04–06) are all LangChain components you assemble.
4. **It's one fixed shape** — the ReAct loop. Custom flow (multi-agent, branching/routing, human-in-the-loop, custom state, persistence/checkpoints) means dropping to **LangGraph** to build the graph yourself.

> `create_agent` = a great shortcut for the common case; LangChain/LangGraph = the toolkit that shortcut is made of, plus everything it doesn't cover.

## Putting it together — `create_agent` + the rest of the toolkit

A support assistant that: searches your docs (RAG tool), does math (custom tool), remembers the conversation (LangGraph memory), and returns structured output. Each part is a different piece of LangChain/LangGraph feeding one agent.

**1. A custom tool** — `@tool` turns a function into a tool (docstring = what the LLM sees):

```python
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression, e.g. '220 * 3'."""
    return str(eval(expression))
```

**2. RAG as a tool** — reuse the vector store from notes 04–06, wrap the retriever as a tool:

```python
from langchain_core.tools import create_retriever_tool

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})     # note 05
docs_tool = create_retriever_tool(
    retriever,
    name="search_docs",
    description="Search the internal product documentation.",
)
```

**3. The model** — provider-agnostic (note 03):

```python
from langchain.chat_models import init_chat_model
model = init_chat_model("openai:gpt-4o-mini", temperature=0)
```

**4. Memory (LangGraph)** — a checkpointer makes the agent remember across turns via a `thread_id`:

```python
from langgraph.checkpoint.memory import InMemorySaver
memory = InMemorySaver()
```

**5. Structured output** — force a typed answer with a Pydantic schema (note 03's `with_structured_output`, here via `response_format`):

```python
from pydantic import BaseModel, Field

class Answer(BaseModel):
    reply: str = Field(description="answer to the user")
    sources_used: list[str] = Field(description="doc sources cited")
```

**6. Assemble + run** — `create_agent` wires the ReAct loop over all of it:

```python
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=[calculator, docs_tool],          # custom tool + RAG tool
    system_prompt="You are a helpful support assistant. Cite sources from the docs.",
    checkpointer=memory,                     # ← memory
    response_format=Answer,                  # ← structured output
)

config = {"configurable": {"thread_id": "user-42"}}   # ties turns to one conversation

r1 = agent.invoke({"messages": [{"role": "user", "content": "What is LangGraph? "}]}, config)
r2 = agent.invoke({"messages": [{"role": "user", "content": "And who created it?"}]}, config)  # remembers r1
print(r2["structured_response"])            # -> Answer(reply=..., sources_used=[...])
```

### What each piece contributed

| Piece | Functionality | From |
|-------|---------------|------|
| `@tool` | custom capability the agent can call | note 07 |
| `create_retriever_tool` + `vectorstore` | RAG (load→split→embed→store→retrieve) as a tool | notes 04–06 |
| `init_chat_model` | swappable model | note 03 |
| `InMemorySaver` + `thread_id` | conversation memory (LangGraph) | LangGraph |
| `response_format=Answer` | typed, validated output | note 03 |
| `create_agent` | the ReAct loop tying it all together | note 07 |

> The agent is the conductor; **the tools, RAG, model, memory, and schema are all separate LangChain/LangGraph parts you build and hand to it.** That's why you learn the whole toolkit, not just `create_agent`.
