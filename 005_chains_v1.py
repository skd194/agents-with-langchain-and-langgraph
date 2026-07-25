"""
    This is a simple demonstration of a basic chain using LangChain. It takes a text input, summarizes it in one sentence using a language model, and prints the result. The code initializes a chat model, defines a prompt template, and sets up a chain that processes the input text through the model and outputs the summary.
"""
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableParallel, 
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch
)

load_dotenv()

model = init_chat_model(model="gpt-4o-mini", model_provider="openai", temperature=0)

def demo_basic_chain():
    prompt = ChatPromptTemplate.from_template("Summarize the following text in one sentence: {text}")
    parser = StrOutputParser()
    chain = prompt | model | parser
    result = chain.invoke(
        {
            "text": "LangChain is a framework for developing applications powered by language models."
        })
    print("Summary:", result)
    
def demo_parallel_chain():
    """Demonstrates a parallel chain that processes multiple inputs simultaneously."""
    
    summarize_prompt = ChatPromptTemplate.from_template("Summarize the following text in one sentence: {text}")
    sentiment_prompt = ChatPromptTemplate.from_template("Analyze the sentiment of the following text: {text}")
    keywords_prompt = ChatPromptTemplate.from_template("Extract 5 keywords from the following text: {text}\n Return them as a comma-separated list.")
    
    parser = StrOutputParser()
    
    # create parallel chain
    analysis_chain = RunnableParallel(
        summary=summarize_prompt | model | parser,
        sentiment=sentiment_prompt | model | parser,
        keywords=keywords_prompt | model | parser,
    )
    
    text = """The LangChain framework simplifies the development of applications that leverage language models,
    providing tools for prompt management, output parsing, and chaining multiple operations together.
    The framework is designed to be flexible and extensible, allowing developers to build complex 
    workflows with ease and efficiency."""
 
    result = analysis_chain.invoke({"text": text})
    print("Analysis Results:")
    for key, value in result.items():
        print(f"\n  {key.capitalize()}: {value}")

def demo_passthrough_chain():
    """Demonstrates a passthrough chain that returns the input without modification."""
    
    prompt = ChatPromptTemplate.from_template(
        "Original question: {question}\n"
        "Context: {context}\n"
        "Answer the question based on the context.")
    

    
    chain = RunnableParallel(
                context=RunnableLambda(fake_retriever),
                question=RunnablePassthrough()
            ) | RunnableLambda(lambda inputs:
                {
                    "question": inputs["question"]["question"],
                    "context": inputs["context"]
                }
            ) | prompt | model | StrOutputParser()
        

    result = chain.invoke({"question": "what is LangChain?"})
    print("Passthrough Result:", result)

def demo_chain_with_branching():
    """Demonstrates a chain with branching logic based on the sentiment of the input text."""
    code_prompt = ChatPromptTemplate.from_template("You are a code expert. Help with: {input}")
    general_prompt = ChatPromptTemplate.from_template("You are a general expert. Answer: {input}")
    
    classifier_prompt = ChatPromptTemplate.from_template(
        "Classify the user's question into exactly one category:\n"
        "- code: about programming, software, or writing/debugging code "
        "(languages, APIs, data structures, algorithms, etc.)\n"
        "- general: anything else.\n\n"
        "Question: {input}\n"
        "Respond with only one word: code or general.")
    
    # temperature=0 -> deterministic label, not the global 0.7 model
    classifier_model = init_chat_model(model="gpt-4o-mini", model_provider="openai", temperature=0)
    classifier = classifier_prompt | classifier_model | StrOutputParser()

    # Branching Chain
    def is_code_question(input_dic):
        classification = classifier.invoke(input_dic)
        print(f"[router] raw classification: {classification!r}")
        return "code" in classification.lower()
    
    branch = RunnableBranch(
        (is_code_question,
         code_prompt | model | StrOutputParser()
         | RunnableLambda(lambda x: {"branch": "code", "answer": x})),
        general_prompt | model | StrOutputParser()  # default branch
        | RunnableLambda(lambda x: {"branch": "general", "answer": x}),
    )

    questions = [
        "write a dictionary in JavaSript",
        "What is the date today?"
    ]

    for q in questions:
        result = branch.invoke({"input": q})
        print(f"Q: {q}")
        print(f"Route: {result['branch']}")
        print(f"A: {result['answer'][:100]}...\n")
        
def demo_debbuging():
    prompt = ChatPromptTemplate.from_template("Say hello to {name}")
    chain = prompt | model | StrOutputParser()

    # Method 1: Get configuration
    print("Chain input schema:", chain.input_schema.model_json_schema())
    print("Chain output schema:", chain.output_schema.model_json_schema())

    # Method 2: Use with_config for tacing
    result = chain.with_config(
        run_name="greeting_chain",
        # tags="demo,debugging",
    ).invoke({"name": "Alice"})
    print(f"Greeting: {result}")

    # Method 3: Inspect intermediate steps
    # Using RunnableLambda for logging
    def log_step(x, step_name=""):
        print(f"[{step_name}] {type(x).__name__}: {str(x)[:100]}")
        return x

    debug_chain = (
        prompt
        | RunnableLambda(lambda x: log_step(x, "after_prompt"))
        | model
        | RunnableLambda(lambda x: log_step(x, "after_model"))
        | StrOutputParser()
    )

    print("\nDebug chain execution:")
    result = debug_chain.invoke({"name": "Debug"})
    print(f"Greeting: {result}")  
    
def fake_retriever(input_dict):
    """# stands in for a real vector-store retriever."""
    return "LangChain is a framework for developing applications powered by language models."

if __name__ == "__main__":
    # demo_basic_chain()
    # demo_parallel_chain()
    # demo_passthrough_chain()
    # demo_chain_with_branching()
    demo_debbuging()