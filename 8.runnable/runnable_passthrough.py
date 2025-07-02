from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()


model = ChatGroq(model = "gemma2-9b-it")


prompt1 = PromptTemplate(
    template = "Write a joke about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Explain the following joke {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    "joke" : RunnablePassthrough(),
    "explanation" : RunnableSequence(prompt2 | model | parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

res = final_chain.invoke({"topic" : "cricket"})

print(res)