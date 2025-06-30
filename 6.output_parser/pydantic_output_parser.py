from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json
from dotenv import load_dotenv
load_dotenv()

llm_endpoint = HuggingFaceEndpoint(
    repo_id= "google/gemma-2-2b-it",
    task =  "text-generation"
)


model = ChatHuggingFace(llm=llm_endpoint)


class Person(BaseModel):

    name : str = Field(description="Name of the person")
    age : int = Field(gt = 18, description= "Age of the person")
    city : str = Field(description="Name of the city the person belong to")


parser = PydanticOutputParser(pydantic_object=Person)


template = PromptTemplate(
    template="Generate the name, age and city of a fictional {place} person. \n {format_instruction}",
    input_variables=["place"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

chain = template | model | parser

final_output = chain.invoke({"place" : "bangladeshi"})

print(type(final_output))