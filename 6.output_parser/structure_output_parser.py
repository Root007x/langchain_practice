from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import json
from dotenv import load_dotenv
load_dotenv()

llm_endpoint = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task =  "text-generation"
)


model = ChatHuggingFace(llm=llm_endpoint)


schema = [
    ResponseSchema(name = "fact_1", description="Fact 1 about the topic"),
    ResponseSchema(name = "fact_2", description="Fact 2 about the topic"),
    ResponseSchema(name = "fact_3", description="Fact 3 about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)


template = PromptTemplate(
    template="Give 3 fact about the {topic} \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({"topic": "black hole"})

print(result)