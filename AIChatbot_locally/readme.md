## Personal AI Chatbot in python

- Ollama - software to run open source llm's
    - download ollama [Ollama](https://ollama.com/)
    - ollama pull llama3
    - ollama run llama3
- Creating virtual environment in python
    - python3 -m venv chatbot
    - source chatbot/bin/activate
    - pip install langchain langchain-ollama ollama

```py
from longchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template="""
    Answer the Question below.
    Here is the conversation history: {context}
    Question: {question}
    Answer:
"""
model = OllamaLLM(model="llama3")
# result = model.invoke(input="hello world")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model
# result = chain.invoke({"context":"", "question":"How are you?"})
# print(result)

def handle_conversation():
    context = ""
    print("Welcome to AI chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You:")
        if user_input.lower() == "exit":
            break
        result = chain.invoke({"context":context, "question":user_input})
        print("Bot: ", result)
        context += f"\nUser:{user_input}\nAI:{result}"

if __name__ = "__main__":
    handle_conversation()

```