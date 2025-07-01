from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


from dotenv import load_dotenv
load_dotenv()


model = ChatGroq(model = "meta-llama/llama-4-scout-17b-16e-instruct")

prompt = PromptTemplate(
    template = "Generate 5 interesting facts about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()


chain = prompt | model | parser

res = chain.invoke({"topic" : "cricket"})

print(res)

chain.get_graph().print_ascii()