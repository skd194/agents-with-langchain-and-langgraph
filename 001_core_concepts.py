from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

load_dotenv()  # Load environment variables from .env file

def demo_basic_chain():
    """
    Demonstrates a basic chain using the LCEL and Runnables.
    """

    # Component 1: Define a prompt template using LCEL
    prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Respond in one sentence to the following input: {input}")

    model = ChatAnthropic(model="claude-haiku-4-5", temperature=0.7)
    parser = StrOutputParser()
    
    # Compose with pipe operator
    chain = prompt | model | parser  # Create a chain by connecting the components
    
    # Execute the chain with an input
    result = chain.invoke({"input": "Why Bharat so special?"})
    
    print("Response:", result)  # Print the result
    
    return chain  # Return the chain for further use if needed
    
def demo_batch_execution():
    """
    Demonstrates batch execution for multiple inputs.
    """
    prompt = ChatPromptTemplate.from_template("Translate to Hindi: {input}")
    model = ChatAnthropic(model="claude-haiku-4-5", temperature=0.7)
    parser = StrOutputParser()
    
    chain = prompt | model | parser  # Create a chain for batch processing
    
    # Batch - run with multiple inputs
    inputs = [
        {"input": "Hello, how are you?"}, 
        {"input": "What is the weather like today?"},
        {"input": "Tell me a joke."}
    ]
    
    results = chain.batch(inputs)  # Execute the chain for all inputs
    
    for text in zip(inputs, results):
        print(f"Input: {text[0]['input']} -> Response: {text[1]}")  # Print input and corresponding response
        
        
def demo_streaming():
    """
    Demonstrates streaming responses from the model.
    """
    prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Respond in 10 sentences to the following input: {input}")
    model = ChatAnthropic(model="claude-haiku-4-5", temperature=0.7, streaming=True)
    parser = StrOutputParser()
    
    chain = prompt | model | parser  # Create a chain for streaming
    
    print("Streaming Response:")
    for chunk in chain.stream({"input": "Explain the concept of quantum computing."}):  # Iterate over the streamed chunks
        print(chunk, end='', flush=True)  # Print each chunk as it arrives

    print()
    
def demo_schema_inspection():
    """
    Demonstrates schema inspection of the chain.
    """
    prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Respond in one sentence to the following input: {input}")
    model = ChatAnthropic(model="claude-haiku-4-5", temperature=0.7)
    parser = StrOutputParser()
    
    chain = prompt | model | parser  # Create a chain
    
    # Inspect the schema of the chain
    input_schema = chain.input_schema.model_json_schema()  # Get the input schema
    output_schema = chain.output_schema.model_json_schema()  # Get the output schema
    
    print("Input Schema:")
    print(input_schema)  # Print the input schema of the chain
    print("Output Schema:")
    print(output_schema)  # Print the output schema of the chain
    
def demo_create_tagline():
    """
    Exercise: Create your first chain using LCEL and Runnables.
    1. Take a product name and target audience as input.
    2. Generate a marketing tagline for the product tailored to the target audience.
    3. Returns the tagline as output.
    
    Test with: 
        Product Name: "EcoClean", Target Audience: "Environmentally conscious consumers"
    """

    prompt = ChatPromptTemplate.from_template("You are a marketing expert. Create a catchy tagline for the product in one sentence: '{product_name}' targeting '{target_audience}'.")
    model = ChatAnthropic(model="claude-haiku-4-5", temperature=0.7)
    parser = StrOutputParser()
    
    chain = prompt | model | parser  # Create a chain for generating marketing taglines
    
    result = chain.invoke({"product_name": "AI Course", "target_audience": "Developers"})  # Execute the chain with the first test input
    
    print("Generated Tagline:", result)
    
def new_way_to_create_chain():
    """
    Universal way to create a chain using LCEL and Runnables.
    """
    model = init_chat_model(
        model_provider="anthropic",
        model="claude-haiku-4-5",
        temperature=0.7,
        max_tokens=1024,
        timeout=60,
        max_retries=3)  # Initialize the chat model
    
    chain = (
        ChatPromptTemplate.from_template("You are a helpful assistant. Respond in one sentence to the following input: {input}")
        | model
        | StrOutputParser()
    )  # Create a chain using the pipe operator
    
    result = chain.invoke({"input": "What is the capital of France?"})  # Execute the chain with an input
    print("Response:", result)  # Print the result

if __name__ == "__main__":
    # demo_basic_chain()  # Run the demo function when the script is executed
    # demo_batch_execution()  # Run the batch execution demo
    # demo_streaming()  # Run the streaming demo
    # demo_schema_inspection()  # Run the schema inspection demo
    # demo_create_tagline()  # Run the exercise for creating a marketing tagline
    new_way_to_create_chain()  # Run the demo for the universal way to create a chain