from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


from dotenv import load_dotenv
load_dotenv()


model = ChatGroq(model = "meta-llama/llama-4-scout-17b-16e-instruct")


prompt_1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)

prompt_2 = PromptTemplate(
    template = "Generate a 5 pointer summary from the following text \n {text}",
    input_variable = ["text"]
)

parser = StrOutputParser()


chain = prompt_1 | model | parser | prompt_2 | model | parser

res = chain.invoke({"topic" : "football"})

print(chain.get_graph().print_ascii())

print(res)
