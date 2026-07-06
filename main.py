"""Simple entrypoint for the agents-with-langchain-and-langgraph demo.

This script is intentionally defensive: it reports which optional
dependencies and environment variables are available so you can
diagnose configuration problems (for example, a mistyped API key
variable in your `.env`).
"""

import sys
from importlib.metadata import version
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from anthropic import Anthropic
import truststore

# Windows consoles default to cp1252, which can't print emoji in
# model responses.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Use the Windows certificate store for TLS verification. Norton's
# Web/Mail Shield re-signs HTTPS traffic with a root certificate that
# is only in the OS store, so certifi-based verification fails.
truststore.inject_into_ssl()


load_dotenv()

langchain_core_version = version("langchain-core")
langgraph_version = version("langgraph")

print(f"LangChain Core version: {langchain_core_version or 'not installed'}")
print(f"LangGraph version: {langgraph_version or 'not installed'}")


def main():
    """Entry point: show env status and run a short LLM test if available."""

    client = Anthropic()
    print("Available Anthropic models:")
    print_anthropic_model_info(client)

    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
    response = llm.invoke("\nSay hello to the world!...\n")
    
    print("\n....LLM test resonse ......")
    print(response.content)
    print(".......")

    print("\nSetup complete...")


def print_anthropic_model_info(client: Anthropic):
    """Print a list of available Anthropic models and their capabilities."""
    for model in anthropic_model_info(client):
        print(f"  {model['id']}: {model['display_name']}")
        print(f"    max input tokens: {model['max_input_tokens']}")
        print(f"    max output tokens: {model['max_tokens']}")
        print(f"    created at: {model['created_at']}")
        print(f"    features: {', '.join(model['features'])}")
        print(".......")

def anthropic_model_info(client: Anthropic):
    """Return a list of available Anthropic models and their capabilities."""
    model_info = []
    for model in client.models.list():
        caps = model.capabilities.model_dump()
        features = []
        if caps["image_input"]["supported"]:
            features.append("vision")
        if caps["thinking"]["types"]["adaptive"]["supported"]:
            features.append("adaptive thinking")
        if caps["structured_outputs"]["supported"]:
            features.append("structured outputs")
        effort_levels = [
            level
            for level in ("low", "medium", "high", "xhigh", "max")
            if caps["effort"].get(level, {}).get("supported")
        ]
        if effort_levels:
            features.append(f"effort: {'/'.join(effort_levels)}")

        model_info.append(
            {
                "id": model.id,
                "display_name": model.display_name,
                "max_input_tokens": model.max_input_tokens,
                "max_tokens": model.max_tokens,
                "created_at": model.created_at,
                "features": features or ["no extended capabilities"],
            }
        )
    return model_info

if __name__ == "__main__":
    main()
