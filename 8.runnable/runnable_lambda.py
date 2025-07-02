from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
load_dotenv()


model = ChatGroq(model = "gemma2-9b-it")

def word_count(text : str):
    return len(text.split())

prompt = PromptTemplate(
    template = "Write a joke about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

seq_chain = RunnableSequence(prompt, model, parser)


parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    "count" : RunnableLambda(word_count)
})

final_chain = RunnableSequence(seq_chain, parallel_chain)

res = final_chain.invoke({
    "topic" : "AI"
})

print(res)
