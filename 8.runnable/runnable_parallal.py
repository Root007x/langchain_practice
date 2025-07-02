from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence
from dotenv import load_dotenv
load_dotenv()


model = ChatGroq(model = "gemma2-9b-it")

prompt1 = PromptTemplate(
    template = "Generate a tweet about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template= "Generate linkedin post about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt1, model, parser),
    'linkedin' : RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({
    "topic" : "AI"
})

print(result)


