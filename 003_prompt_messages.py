from dotenv import load_dotenv
from langchain_core import messages
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.prompts import FewShotChatMessagePromptTemplate

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ChatMessage,
    ToolMessage,
)


load_dotenv()  # Load environment variables from .env file

prompt = ChatPromptTemplate.from_template("Tell me a {adjective} story about {topic}")

messages = prompt.format_messages(adjective="funny", topic="cat")

print(messages)  # Print the formatted message

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
        ("human", "Translate the following text: {text}")
    ]
)

messages = prompt.format_messages(input_language="English", output_language="Hindi", text="How are you?")
model  = init_chat_model(
    model_provider="anthropic",
    model="claude-haiku-4-5",
    temperature=0.7,
    max_tokens=1024,
    timeout=60,
    max_retries=3)  # Initialize the chat model

response = model.invoke(messages)
print(response.content)  # Print the model's response

messages = [
    SystemMessage(content="You are a helpful assistant that translates English to Hindi."),
    HumanMessage(content="Translate the following text: How are you?"),
    AIMessage(content="This is an AI message."),
]


# Few-shot learning example
print("\nFew-shot learning example:")

examples = [
    {
        "input": "I absolutely love this phone. The battery life is amazing.",
        "output": "Positive"
    },
    {
        "input": "The service was terrible. I will never come back.",
        "output": "Negative"
    },
    {
        "input": "The package arrived yesterday.",
        "output": "Neutral"
    },
    {
        "input": "This laptop is fast and lightweight.",
        "output": "Positive"
    },
    {
        "input": "The food was cold and tasteless.",
        "output": "Negative"
    },
]

# -----------------------------------------------------
# Template for each example
# -----------------------------------------------------

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)

# -----------------------------------------------------
# Final Prompt
# -----------------------------------------------------

final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a sentiment analysis assistant.

Classify the sentiment of the user's text into exactly one of these labels:

- Positive
- Negative
- Neutral

Return ONLY the label.
Do not explain your answer.
            """,
        ),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

# -----------------------------------------------------
# Initialize Claude
# -----------------------------------------------------

model = init_chat_model(
    model_provider="anthropic",
    model="claude-haiku-4-5",
    temperature=0,
    max_tokens=20,
    timeout=60,
    max_retries=3,
)

# -----------------------------------------------------
# User input
# -----------------------------------------------------

messages = final_prompt.format_messages(input="The movie was fantastic. I enjoyed every minute.")
response = model.invoke(messages)

print(response.content)