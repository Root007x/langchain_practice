from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
from dotenv import load_dotenv
load_dotenv()

llm_endpoint = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task =  "text-generation"
)


model = ChatHuggingFace(llm=llm_endpoint)

parser = JsonOutputParser()

template = PromptTemplate(
    template = "Give me the name, age and city of a fictional person. \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# option 1
# prompt = template.format()

# result = model.invoke(prompt)


# final_result = parser.parse(result.content)
# print(type(final_result))

# option 2
chain = template | model | parser

result = chain.invoke({})

print(result)

# with open("json_output.json", "w") as f:
#     json.dump(final_result, f)