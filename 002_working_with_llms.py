from dotenv import load_dotenv
from langchain_core import messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv()  # Load environment variables from .env file

def demo_messages():
    
    model = init_chat_model(
        model_provider="anthropic",
        model="claude-haiku-4-5",
        temperature=0.7,
        max_tokens=1024,
        timeout=60,
        max_retries=3)  # Initialize the chat model
    
    messages = [
        SystemMessage(
            content="You are a software engineer. Give concise, technically accurate answers."
        ),
        HumanMessage(content="Explain what REST APIs are.")
    ]

    response = model.invoke(messages)
   
    messages.append(response)
    messages.append(HumanMessage(content="Can you give me a simple C# example?"))
    
    response = model.invoke(messages)
    
    print(messages)
    print()
    print("Response:", response)  # Print the model's response


def multi_model_setup():
    model1 = init_chat_model(
        model_provider="anthropic",
        model="claude-haiku-4-5",
        temperature=0.7,
        max_tokens=1024,
        timeout=60,
        max_retries=3)  # Initialize the first chat model

    model2 = init_chat_model(
        model_provider="anthropic",
        model="claude-sonnet-4-5",
        temperature=0.7,
        max_tokens=1024,
        timeout=60,
        max_retries=3)  # Initialize the second chat model

    messages = [
        SystemMessage(
            content="You are a software engineer. Give concise, technically accurate answers."
        ),
        HumanMessage(content="Explain what REST APIs are.")
    ]

    response1 = model1.invoke(messages)
    response2 = model2.invoke(messages)
    
    # # print dictionary of model and response
    
    model1 = response1.response_metadata["model_name"]
    model2 = response2.response_metadata["model_name"]
    model_responses = {
        model1: response1.content,
        model2: response2.content
    }
    
    print(model_responses)


if __name__ == "__main__":
    # demo_messages()  # Run the demo if this script is executed directly
    multi_model_setup()  # Run the multi-model setup if this script is executed directly    