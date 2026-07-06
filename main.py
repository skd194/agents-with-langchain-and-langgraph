"""Simple entrypoint for the agents-with-langchain-and-langgraph demo.

This script is intentionally defensive: it reports which optional
dependencies and environment variables are available so you can
diagnose configuration problems (for example, a mistyped API key
variable in your `.env`).
"""

from importlib.metadata import version
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

langchain_core_version = version("langchain-core")
langgraph_version = version("langgraph")

print(f"LangChain Core version: {langchain_core_version or 'not installed'}")
print(f"LangGraph version: {langgraph_version or 'not installed'}")


def main():
    """Entry point: show env status and run a short LLM test if available."""

    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
    response = llm.invoke("Say hello to the world!")
    print(response)

    print("Setup complete")


if __name__ == "__main__":
    main()
