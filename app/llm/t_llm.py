# main.py
from llm import LLMClient

def run():
    llm = LLMClient()
    question = "Explain LangChain in simple terms."
    answer = llm.chat(question)
    print("Q:", question)
    print("A:", answer)

if __name__ == "__main__":
    run()
