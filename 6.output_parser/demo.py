from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
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

prompt_1 = template_1.invoke({"topic": "Black Hole"})

result_1 = model.invoke(prompt_1)

prompt_2 = template_2.invoke({"text": result_1.content})

result_2 = model.invoke(prompt_2)

print(result_2.content)