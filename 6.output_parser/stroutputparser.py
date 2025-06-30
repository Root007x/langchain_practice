from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

llm_endpoint = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task =  "text-generation"
)


model = ChatHuggingFace(llm=llm_endpoint)


template_1 = PromptTemplate(
    template= "Write a detailed report on {topic}",
    input_variables=["topic"]
)


template_2 = PromptTemplate(
    template= "Write a 5 line summary on the following text. \n {text}",
    input_variables=["text"]
)


parser = StrOutputParser()

chain = template_1 | model | parser | template_2 | model | parser

result = chain.invoke({
    'topic' : "black hole"
})

print(result)